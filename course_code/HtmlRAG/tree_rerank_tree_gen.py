import argparse
import json
import os
import re
import threading

import bs4
import loguru
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import traceback
import sys

sys.path.append("./")
from html4rag.html_utils import trim_html_tree

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dataset", type=str, default="kdd_crag")
    argparser.add_argument("--split", type=str, default="test")
    argparser.add_argument("--ckpt_path", type=str, default="../../huggingface/glm-4-9b-chat-1m")
    argparser.add_argument("--ckpt_version", type=str, default="zeroshot")
    argparser.add_argument("--context_window", type=str, default="2k")
    argparser.add_argument("--chat_tokenizer_name", type=str, default="Llama-3.2-3B-Instruct")
    argparser.add_argument("--fine_trim_ratio", type=str, default="1/2")
    argparser.add_argument("--src_max_node_words", type=int, default=256)
    argparser.add_argument("--max_node_words", type=int, default=128)
    args = argparser.parse_args()
    ckpt_path = args.ckpt_path
    split = args.split
    dataset = args.dataset
    ckpt_version = args.ckpt_version
    context_window = args.context_window
    chat_tokenizer_name = args.chat_tokenizer_name
    fine_trim_ratio = args.fine_trim_ratio
    src_max_node_words = args.src_max_node_words
    max_node_words = args.max_node_words

    loguru.logger.info(f"ckpt version: {ckpt_version}, path: {ckpt_path}")
    loguru.logger.info(f"max node words: {max_node_words}")
    assert ckpt_version in ckpt_path, f"ckpt version {ckpt_version} mismatch with ckpt path {ckpt_path}"
    max_context_window = re.match(r"(\d+)k", context_window).group(1)
    max_context_window = int(max_context_window) * 1000
    #  remove low prob paths pointed tags
    loguru.logger.info(f"trimming htmls with context window {context_window}, max node words {max_node_words}")

    # ckpt_path="/cpfs01/shared/public/guopeidong/models/glm4-9b/glm4-9b-128k-v0701-node2/checkpoint-1554"
    node_tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
    if chat_tokenizer_name == "bc":
        chat_tokenizer_path = "../../huggingface/Baichuan2-7B-Chat/"
    elif chat_tokenizer_name == "llama":
        chat_tokenizer_path = "../../huggingface/Meta-Llama-3.1-70B-Instruct/"
    else:
        raise ValueError(f"chat_tokenizer_name {chat_tokenizer_name} not supported")
    chat_tokenizer = AutoTokenizer.from_pretrained(chat_tokenizer_path, trust_remote_code=True)
    loguru.logger.info(f"node tokenizer: {node_tokenizer.name_or_path}, chat tokenizer: {chat_tokenizer.name_or_path}")

    thread_pool = []
    if fine_trim_ratio == "custom":
        if dataset in ["asqa", "nq", "eli5"]:
            fine_trim_ratio = "2/3"
        else:
            fine_trim_ratio = "1/2"
    if fine_trim_ratio == "1/2":
        coarse_context_window = \
            {"1k": "2k", "2k": "4k", "4k": "8k", "8k": "16k", "16k": "32k", "32k": "64k", "64k": "128k"}[context_window]
    elif fine_trim_ratio == "2/3":
        coarse_context_window = \
            {"2k": "3k", "4k": "6k", "8k": "12k", "16k": "24k", "32k": "48k", "64k": "96k"}[context_window]
    else:
        raise ValueError(f"fine_trim_ratio {fine_trim_ratio} not supported")
    data_file = f"./html_data/{dataset}/tree-rerank/{chat_tokenizer_name}/{src_max_node_words}-{dataset}-{split}-{coarse_context_window}.jsonl"
    data_lines = [json.loads(line) for line in open(data_file)]

    if torch.cuda.is_available():
        device = "cuda"
        parallel_size = torch.cuda.device_count()
        loguru.logger.info(f"Parallel size: {parallel_size}")
        shard_pool = []
    else:
        # model=AutoModelForCausalLM.from_pretrained("../../../huggingface/glm-4-9b-chat-1m",trust_remote_code=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(ckpt_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
        model.max_node_words = max_node_words
        device = "cpu"
        model.to(device).eval()


    def init_shard_model(rank):
        shard_model = AutoModelForSeq2SeqLM.from_pretrained(ckpt_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
        shard_model.max_node_words = max_node_words
        shard_model.to(f"cuda:{rank}").eval()
        shard_pool.append(shard_model)


    #  copy model to all devices
    if device == "cuda" and parallel_size > 1:
        for rank in range(parallel_size):
            thread = threading.Thread(target=init_shard_model, args=(rank,))
            thread.start()
            thread_pool.append(thread)
        for thread in thread_pool:
            thread.join()

    loguru.logger.info(f"Reading data from {data_file}")
    if args.mini_dataset:
        data_lines = data_lines[:10]

    total_len = len(data_lines)
    res_lines = [{} for _ in range(total_len)]
    pbar = tqdm(total=total_len, desc=f"Processing {dataset} {split}")

    output_file = f"./html_data/{dataset}/tree-rerank-tree-gen/{ckpt_version}/{chat_tokenizer_name}/{src_max_node_words}to{max_node_words}-{dataset}-{split}-{coarse_context_window}to{context_window}.jsonl"
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)


    def start_thread(rank):
        while len(data_lines) > 0:
            try:
                idx = total_len - len(data_lines)
                data_line = data_lines.pop(0)
                question = data_line['question']
                coarse_html_trim = data_line["html_trim"]
                html_res = shard_pool[rank].generate_html_tree(node_tokenizer, [question], [coarse_html_trim])
                # loguru.logger.info(f"Calculate probs for {len(html_res[0]['path_probs'])} nodes")
                data_line.pop(f'search_results', None)

                res_lines[idx] = {**data_line, **html_res[0]}
                res_lines[idx]["html_trim"] = trim_html_tree(
                    html=html_res[0]["html"],
                    paths=html_res[0]["paths"],
                    is_leaf=html_res[0]["is_leaf"],
                    node_tree=html_res[0]["node_tree"],
                    chat_tokenizer=chat_tokenizer,
                    node_tokenizer=node_tokenizer,
                    max_context_window=max_context_window,
                )

            except Exception as e:
                loguru.logger.error(f"Error in processing line {idx}: {e}")
                traceback.print_exc()
                # print(f"Error in processing line {idx}: {e}")
                #  save the processed data
                with open(output_file, "w") as f:
                    for idx in range(len(res_lines)):
                        #  convert "path_probs" from float32 to string
                        # res_lines[idx]["path_probs"] = [str(prob) for prob in res_lines[idx]["path_probs"]]
                        try:
                            f.write(json.dumps(res_lines[idx], ensure_ascii=False) + "\n")
                        except Exception as e:
                            # loguru.logger.error(f"Error in writing line {idx}: {e}")
                            f.write(json.dumps(res_lines[idx], ensure_ascii=True) + "\n")
            pbar.update(1)


    for i in range(len(shard_pool)):
        thread = threading.Thread(target=start_thread, args=(i,))
        thread.start()
        thread_pool.append(thread)

    for thread in thread_pool:
        thread.join()

    pbar.close()

    with open(output_file, "w") as f:
        for idx in range(len(res_lines)):
            #  convert "path_probs" from float32 to string
            # res_lines[idx]["path_probs"] = [str(prob) for prob in res_lines[idx]["path_probs"]]
            try:
                f.write(json.dumps(res_lines[idx], ensure_ascii=False) + "\n")
            except Exception as e:
                loguru.logger.error(f"Error in writing line {idx}: {e}")
                f.write(json.dumps(res_lines[idx], ensure_ascii=True) + "\n")
    loguru.logger.info(f"Saved parsed html to {output_file}")
