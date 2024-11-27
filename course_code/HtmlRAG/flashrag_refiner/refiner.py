import argparse
import json
import os
import threading
from typing import List

import loguru
from bs4 import BeautifulSoup
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm
import re
import torch
import numpy as np

import sys

sys.path.append("./")
from html4rag.chunk_rerank import html_splitter
from html4rag.html_utils import clean_xml
from flashrag_refiner.utils import load_model, pooling
from flashrag_refiner.dataset import Item, Dataset


class BaseRefiner:
    r"""Base object of Refiner method"""

    def __init__(self, config):
        self.config = config
        self.name = config["refiner_name"]
        self.model_path = config["refiner_model_path"]
        self.device = config["device"]
        self.input_prompt_flag = config["refiner_input_prompt_flag"] if "refiner_input_prompt_flag" in config else False

    def run(self, item) -> str:
        r"""Get refining result.

        Args:
            item: dataset item, contains question, retrieval result...

        Returns:
            str: refining result of this item
        """
        pass

    def batch_run(self, dataset, batch_size=None) -> List[str]:
        return [self.run(item) for item in dataset]


class LLMLinguaRefiner(BaseRefiner):
    """Implementation for (Long)LLMLingua."""

    def __init__(self, config):
        super().__init__(config)
        default_config = {
            "rate": 0.55,
            "condition_in_question": "after_condition",
            "reorder_context": "sort",
            "dynamic_context_compression_ratio": 0.3,
            "condition_compare": True,
            "context_budget": "+100",
            "rank_method": "longllmlingua",
        }
        if "llmlingua_config" in config and config["llmlingua_config"] is not None:
            self.compress_config = config["llmlingua_config"]
        else:
            self.compress_config = default_config

        from flashrag_refiner.llmlingua_compressor import PromptCompressor

        self.refiner = PromptCompressor(model_name=self.model_path, device_map=config["device"])

    def format_reference(self, retrieval_result):
        format_reference = ""
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item["contents"]
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx + 1}(Title: {title}) {text}\n"

        return format_reference

    def batch_run(self, dataset):
        output = []
        for item in dataset:
            question = item["question"]
            retrieval_result = item["retrieval_result"]
            # TODO: suit more cases
            if self.input_prompt_flag:
                input_prompt = item["prompt"]
                prompt_split = input_prompt.split("\n\n")
                # need fixed format prompt: instr + demon(retrieval results) + question
                instruction, question = prompt_split[0], prompt_split[-1]
                demonstration = "\n".join(prompt_split[1:-1])
                item_output = self.refiner.compress_prompt(
                    [i for i in demonstration.split("\n") if i != ""],
                    instruction=instruction,
                    question=question,
                    **self.compress_config,
                )
            else:
                docs = self.format_reference(retrieval_result).split("\n")
                docs = [i for i in docs if i != ""]
                item_output = self.refiner.compress_prompt(
                    docs, instruction="", question=question, **self.compress_config
                )
            output.append(item_output["compressed_prompt"])
        return output


class SelectiveContextRefiner(BaseRefiner):
    """Implementation for Selective Context"""

    def __init__(self, config):
        super().__init__(config)
        from flashrag_refiner.selective_context_compressor import SelectiveContext

        default_config = {"reduce_ratio": 0.5, "target_token": 2000}

        self.refiner = SelectiveContext(model_type="gpt2", model_path=self.model_path, lang="en")
        if "sc_config" in config and config["sc_config"] is not None:
            self.compress_config = config["sc_config"]
        else:
            self.compress_config = default_config

    def format_reference(self, retrieval_result):
        format_reference = ""
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item["contents"]
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx + 1}(Title: {title}) {text}\n"

        return format_reference

    def batch_run(self, dataset):
        # only use text
        all_inputs = []
        for item in dataset:
            retrieval_result = item["retrieval_result"]
            all_inputs.append(self.format_reference(retrieval_result))

        output = []
        for text in all_inputs:
            compress_text, _ = self.refiner(text, **self.compress_config)
            output.append(compress_text)
        return output


class ExtractiveRefiner(BaseRefiner):
    """Implementation for Extractive compressor.
    Using retrieval method to select sentences or other granularity data.
    """

    def __init__(self, config):
        super().__init__(config)
        # number of keeping sentences
        self.topk = config["refiner_topk"]
        self.pooling_method = config["refiner_pooling_method"]

        self.encode_max_length = config["refiner_encode_max_length"]

        # load model
        self.encoder, self.tokenizer = load_model(self.model_path, use_fp16=True)

    def encode(self, query_list: List[str], is_query=True):
        if "e5" in self.model_path.lower():
            if is_query:
                query_list = [f"query: {query}" for query in query_list]
            else:
                query_list = [f"passage: {query}" for query in query_list]

        inputs = self.tokenizer(
            query_list, max_length=self.encode_max_length, padding=True, truncation=True, return_tensors="pt"
        )
        inputs = {k: v.cuda() for k, v in inputs.items()}

        if "T5" in type(self.encoder).__name__:
            # T5-based retrieval model
            decoder_input_ids = torch.zeros((inputs["input_ids"].shape[0], 1), dtype=torch.long).to(
                inputs["input_ids"].device
            )
            output = self.encoder(**inputs, decoder_input_ids=decoder_input_ids, return_dict=True)
            query_emb = output.last_hidden_state[:, 0, :]

        else:
            output = self.encoder(**inputs, return_dict=True)
            query_emb = pooling(
                output.pooler_output, output.last_hidden_state, inputs["attention_mask"], self.pooling_method
            )
            if "dpr" not in self.model_path.lower():
                query_emb = torch.nn.functional.normalize(query_emb, dim=-1)

        query_emb = query_emb.detach().cpu().numpy()
        query_emb = query_emb.astype(np.float32)
        return query_emb

    def batch_run(self, dataset, batch_size=16):
        questions = [i["question"] for i in dataset]
        # only use text
        retrieval_results = [i["retrieval_result"] for i in dataset]
        retrieval_results = [
            ["\n".join(doc_item["contents"].split("\n")[1:]) for doc_item in item_result]
            for item_result in retrieval_results
        ]

        # split into sentences: [[sent1, sent2,...], [...]]
        sent_lists = [
            [i.strip() for i in re.split(r"(?<=[.!?])\s+", " ".join(res)) if len(i.strip()) > 5]
            for res in retrieval_results
        ]
        score_lists = []  # matching scores, size == sent_lists
        for idx in range(0, len(questions), batch_size):
            batch_questions = questions[idx: idx + batch_size]
            batch_sents = sent_lists[idx: idx + batch_size]

            question_embs = self.encode(batch_questions, is_query=True)
            sent_embs = self.encode(sum(batch_sents, []), is_query=False)  # n*d
            scores = question_embs @ sent_embs.T
            start_idx = 0
            for row_score, single_list in zip(scores, sent_lists):
                row_score = row_score.tolist()
                score_lists.append(row_score[start_idx: start_idx + len(single_list)])
                start_idx += len(single_list)

        # select topk sents
        retain_lists = []
        for sent_scores, sent_list in zip(score_lists, sent_lists):
            if len(sent_scores) < self.topk:
                retain_lists.append(sent_list)
                continue
            topk_idxs = torch.topk(torch.Tensor(sent_scores), self.topk).indices.tolist()
            retain_lists.append([sent_list[idx] for idx in sorted(topk_idxs)])

        return [" ".join(sents) for sents in retain_lists]


class AbstractiveRecompRefiner(BaseRefiner):
    """Implementation for Abstractive RECOMP compressor:
    RECOMP: Improving Retrieval-Augmented LMs with Compression and Selective Augmentation.
    """

    def __init__(self, config):
        super().__init__(config)

        self.max_input_length = config["refiner_max_input_length"]
        self.max_output_length = config["refiner_max_output_length"]

        # load model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
        self.model.cuda()
        self.model.eval()

    def batch_run(self, dataset, batch_size=2):
        # only use text
        retrieval_results = [i["retrieval_result"] for i in dataset]
        retrieval_results = [
            ["\n".join(doc_item["contents"].split("\n")[1:]) for doc_item in item_result]
            for item_result in retrieval_results
        ]

        # input processing in recomp training format
        format_inputs = [
            "Question: {question}\n Document: {document}\n Summary: ".format(
                question=item["question"], document="\n".join(docs)
            )
            for item, docs in zip(dataset, retrieval_results)
        ]

        results = []
        for idx in range(0, len(format_inputs), batch_size):
            batch_inputs = format_inputs[idx: idx + batch_size]
            batch_inputs = self.tokenizer(
                batch_inputs, return_tensors="pt", padding=True, truncation=True, max_length=self.max_input_length
            ).to(self.device)

            batch_outputs = self.model.generate(**batch_inputs, max_length=self.max_output_length)

            batch_outputs = self.tokenizer.batch_decode(
                batch_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            results.extend(batch_outputs)

        return results


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--chat_model", type=str, default="bc33b192k")
    argparser.add_argument("--dataset", type=str, default="wstqa")
    argparser.add_argument("--split", type=str, default="test")
    argparser.add_argument("--search_engine", type=str, default="bing")
    argparser.add_argument("--mini_dataset", action="store_true")
    argparser.add_argument("--rewrite_method", type=str, default="slimplmqr")
    argparser.add_argument("--refiner_name", type=str, default="llmlingua")
    argparser.add_argument("--context_window", type=str, default="2k")
    argparser.add_argument("--chat_tokenizer_name", type=str, default="llama")
    args = argparser.parse_args()

    dataset = args.dataset
    split = args.split
    search_engine = args.search_engine
    mini_dataset = args.mini_dataset
    rewrite_method = args.rewrite_method
    refiner_name = args.refiner_name
    context_window = args.context_window
    chat_tokenizer_name = args.chat_tokenizer_name

    max_context_window = int(context_window[:-1]) * 1000

    if dataset in ["wstqa", "statsgov"]:
        language = "zh"
        search_engine = ""
        loguru.logger.info(f"setting language to {language}, search_engine to empty for dataset {dataset}")
    elif dataset in ["websrc", "asqa", "nq", "hotpot-qa", "trivia-qa", "eli5", "musique"]:
        language = "en"
        corpus = ""
        loguru.logger.info(f"setting language to {language}, corpus to empty for dataset {dataset}")
    else:
        raise NotImplementedError(f"dataset {dataset} not implemented")

    assert search_engine == "" or corpus == "", "search_engine and corpus cannot be both set"
    if search_engine in ["bing", "google"]:
        # multi docs
        if rewrite_method == "vanilla_search":
            input_file = f"./html_data/{dataset}/{search_engine}/{search_engine}html-{dataset}-{split}.jsonl"
            output_file = f"./html_data/{dataset}/baselines/{chat_tokenizer_name}/{search_engine}html-{refiner_name}-{dataset}-{split}-{context_window}.jsonl"
        elif rewrite_method in ["slimplmqr", ]:
            input_file = f"./html_data/{dataset}/{search_engine}/{search_engine}html-{rewrite_method}-{dataset}-simple-{split}.jsonl"
            output_file = f"./html_data/{dataset}/baselines/{chat_tokenizer_name}/{search_engine}html-{rewrite_method}-{refiner_name}-{dataset}-{split}-{context_window}.jsonl"
        else:
            raise NotImplementedError(f"rewrite method {rewrite_method} not implemented")
        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file))
    else:
        raise NotImplementedError(f"search engine {search_engine} not implemented")

    loguru.logger.info(f"input_file: {input_file}")
    data_lines = [json.loads(l) for l in open(input_file)]
    if args.mini_dataset:
        data_lines = data_lines[:10]

    flashrag_dataset = []
    for data_line in data_lines:
        retrieval_results = []
        for res in data_line[f"{rewrite_method}_results"]:
            html = clean_xml(res["html"]).strip()
            if html:
                soup = BeautifulSoup(html, 'html.parser')
                if soup.title:
                    title = soup.title.string
                else:
                    title = ""
                splits = [" ".join(d.page_content.split("\n")) for d in html_splitter.split_text(html)]
                text = "\n".join(splits)
                html = f"{title}\n{text}"
                retrieval_results.append({"contents": html})
        flashrag_dataset.append(
            {
                "question": data_line["question"],
                "retrieval_result": retrieval_results,
            }
        )

    # init refiners on different devices
    refiner_pool = []
    thread_pool = []


    def init_refiner(device):
        if refiner_name == "llmlingua":
            config = {
                "refiner_name": "llmlingua",
                "refiner_model_path": "../../huggingface/llama-2-7b-chat-hf",
                "device": device,
                "refiner_input_prompt_flag": False,
                "llmlingua_config": {
                    "target_token": max_context_window,
                    "condition_in_question": "after_condition",
                    "reorder_context": "sort",
                    "dynamic_context_compression_ratio": 0.3,
                    "condition_compare": True,
                    "context_budget": "+100",
                    "rank_method": "longllmlingua",
                },
            }
            refiner = LLMLinguaRefiner(config)
            refiner_pool.append(refiner)
        elif refiner_name == "selective-context":
            config = {
                "refiner_name": "selective-context",
                "refiner_model_path": "../../huggingface/gpt2",
                "device": device,
                "refiner_input_prompt_flag": False,
                "sc_config": {"target_token": max_context_window},
            }
            refiner = SelectiveContextRefiner(config)
            refiner_pool.append(refiner)
        else:
            raise NotImplementedError(f"refiner {refiner_name} not implemented")


    if torch.cuda.is_available() and torch.cuda.device_count() >= 1:
        device_list = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        for device in device_list:
            thread = threading.Thread(target=init_refiner, args=(device,))
            thread.start()
            thread_pool.append(thread)
    else:
        raise NotImplementedError("gpu not available")

    for thread in thread_pool:
        thread.join()

    thread_pool = []

    total_len = len(flashrag_dataset)
    res_lines = [{} for _ in range(total_len)]
    loguru.logger.info(f"Total number of nodes: {total_len}")
    pbar = tqdm(total=total_len, desc=f"Processing {dataset} {context_window}")


    def start_thread(rank):
        while len(flashrag_dataset) > 0:
            try:
                idx = total_len - len(flashrag_dataset)
                item = flashrag_dataset.pop(0)
                html_trim = refiner_pool[rank].batch_run([item])[0]
                res_lines[idx] = {**data_lines[idx], "html_trim": html_trim}

            except Exception as e:
                print(e)
            pbar.update(1)


    for i in range(len(refiner_pool)):
        thread = threading.Thread(target=start_thread, args=(i,))
        thread.start()
        thread_pool.append(thread)

    for thread in thread_pool:
        thread.join()

    pbar.close()

    with open(output_file, "w") as f:
        for line in res_lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    loguru.logger.info(f"output_file: {output_file}")
