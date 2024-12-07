import bz2
import argparse
import bz2
import concurrent.futures
import json
import multiprocessing
import os
import re
import sys
import torch
import loguru
from bs4 import BeautifulSoup
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import HTMLHeaderTextSplitter, RecursiveCharacterTextSplitter
from loguru import logger
from tqdm import tqdm
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import bs4
from html4rag.html_utils import split_tree, clean_xml, simplify_html
sys.path.append("./")
from html4rag.html_utils import trim_html_tree, HTMLSplitter, headers_to_split_on, clean_xml, truncate_input, trim_path
from langchain_core.documents import Document
import traceback
import threading
from htmlrag import GenHTMLPruner

def load_data_in_batches(dataset_path, batch_size, split=-1):
    """
    Generator function that reads data from a compressed file and yields batches of data.
    Each batch is a dictionary containing lists of interaction_ids, queries, search results, query times, and answers.

    Args:
    dataset_path (str): Path to the dataset file.
    batch_size (int): Number of data items in each batch.

    Yields:
    dict: A batch of data.
    """

    def initialize_batch():
        """ Helper function to create an empty batch. """
        return {"interaction_id": [], "query": [], "search_results": [], "query_time": [], "answer": []}

    try:
        with bz2.open(dataset_path, "rt") as file:
            batch = initialize_batch()
            for line in file:
                try:
                    item = json.loads(line)

                    if split != -1 and item["split"] != split:
                        continue

                    for key in batch:
                        batch[key].append(item[key])

                    if len(batch["query"]) == batch_size:
                        yield batch
                        batch = initialize_batch()
                except json.JSONDecodeError:
                    logger.warn("Warning: Failed to decode a line.")
            # Yield any remaining data as the last batch
            if batch["query"]:
                yield batch
    except FileNotFoundError as e:
        logger.error(f"Error: The file {dataset_path} was not found.")
        raise e
    except IOError as e:
        logger.error(f"Error: An error occurred while reading the file {dataset_path}.")
        raise e


def generate_predictions(dataset_path, split):
    """
    Processes batches of data from a dataset to generate predictions using a model.

    Args:
    dataset_path (str): Path to the dataset.
    model (object): UserModel that provides `get_batch_size()` and `batch_generate_answer()` interfaces.

    Returns:
    tuple: A tuple containing lists of queries, ground truths, and predictions.
    """
    interaction_id, query_times, queries, answers, search_results = [], [], [], [], []
    # batch_size = model.get_batch_size()

    for batch in tqdm(load_data_in_batches(dataset_path, 1, split), desc="Generating Simplify HTML"):
        # batch_ground_truths = batch.pop("answer")  # Remove answers from batch and store them
        interaction_id.extend(batch["interaction_id"])
        queries.extend(batch["query"])
        answers.extend(batch["answer"])
        query_times.extend(batch["query_time"])
        cur_search_results = []
        for search in batch["search_results"][0]:
            html_source = search["page_result"]
            h_soup = BeautifulSoup(html_source, 'html.parser')
            html_source = simplify_html(h_soup, keep_attr=True)
            cur_search_results.append(html_source)
        search_results.append(cur_search_results)
    return interaction_id, query_times, queries, answers, search_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path", type=str, default="../data/crag_task_1_dev_v4_release.jsonl.bz2",
                        choices=["../example_data/dev_data.jsonl.bz2",  # example data
                                 "../data/crag_task_1_dev_v4_release.jsonl.bz2",  # full data
                                 ])
    parser.add_argument("--split", type=int, default=1,
                        help="The split of the dataset to use. This is only relevant for the full data: "
                             "0 for public validation set, 1 for public test set")

    parser.add_argument("--llm_name", type=str, default="../pretrained_model/meta-llama/Llama-3.2-3B-Instruct",
                        choices=["../pretrained_model/meta-llama/Llama-3.2-3B-Instruct",
                                 "../pretrained_model/google/gemma-2-2b-it",
                                 # can add more llm models here
                                 ])
    parser.add_argument("--model_name", type=str, default="htmlrag",
                        choices=["vanilla_baseline",
                                 "rag_baseline",
                                 "multifeature",
                                 "htmlrag",
                                 "bge_baseline",
                                 "bm25"
                                 # add your model here
                                 ],
                        )
    parser.add_argument("--url", type=str, default="../../pretrained_model/BAAI/bge-large-en")
    parser.add_argument("--device", type=str, default="cuda:2")
    parser.add_argument("--context_window", type=str, default="4k")
    parser.add_argument("--chunk_size", type=int, default=1000)
    parser.add_argument("--max_node_words", type=int, default=1000)
    args = parser.parse_args()
    context_window = args.context_window
    model_name = args.model_name
    url = args.url
    dataset_path = args.dataset_path
    dataset = dataset_path.split("/")[1]
    llm_name = args.llm_name
    _llm_name = llm_name.split("/")[-1]
    max_node_words = args.max_node_words
    split = -1
    if dataset == "data":
        split = args.split
        if split == -1:
            raise ValueError("Please provide a valid split value for the full data: "
                             "0 for public validation set, 1 for public test set.")
    dataset_path = os.path.join("..", dataset_path)
    html_splitter = HTMLSplitter(headers_to_split_on=headers_to_split_on)
    # html_splitter = HTMLHeaderTextSplitter(headers_to_split_on)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size, chunk_overlap=100
    )
    # Generate predictions
    interaction_id, query_times, queries, answers, search_results = generate_predictions(dataset_path, split)

    # Combine results into a JSON-serializable structure
    data_lines = [
        {
            "id": interaction_id[i],
            "query_time": query_times[i],
            "question": queries[i],
            "answers": [answers[i]],
            "search_results": search_results[i]
        }
        for i in range(len(interaction_id))
    ]



    thread_pool = []
    rerank_model = "bgelargeen"
    query_instruction_for_retrieval = "Represent this sentence for searching relevant passages: "

    rerank_model_name = "../../pretrained_model/BAAI/bge-large-en"
    tokenizer=AutoTokenizer.from_pretrained(rerank_model_name, trust_remote_code=True)

    embedder = HuggingFaceBgeEmbeddings(
        model_name=url,
        model_kwargs={'device': args.device},
        encode_kwargs={'normalize_embeddings': True},
    )

    # embedder = TEIEmbeddings(
    #     model=args.url,
    #     huggingfacehub_api_token="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",
    #     model_kwargs={"truncate": True})


    # with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    for nidx in tqdm(range(len(data_lines)), total=len(data_lines), desc=f"{dataset} ranking:"):
        htmls = [clean_xml(d) for d in data_lines[nidx][f'search_results']]
        htmls = [h for h in htmls if h.strip()]
        soup = bs4.BeautifulSoup("", 'html.parser')
        for html in htmls:
            soup.append(bs4.BeautifulSoup(html, 'html.parser'))
        split_res = split_tree(soup, max_node_words=max_node_words)
        path_tags = [res[0] for res in split_res]
        paths = [res[1] for res in split_res]
        is_leaf = [res[2] for res in split_res]
        question = query_instruction_for_retrieval + data_lines[nidx]['question']

        node_docs = []
        for pidx in range(len(paths)):
            node_docs.append(Document(page_content=path_tags[pidx].get_text(), metadata={"path_idx": pidx}))
        batch_size = 256

        db = FAISS.from_documents(node_docs[:batch_size], embedder)
        if len(node_docs) > batch_size:
            for doc_batch_idx in range(batch_size, len(node_docs), batch_size):
                db.add_documents(node_docs[doc_batch_idx:doc_batch_idx + batch_size])

        retriever = db.as_retriever(search_kwargs={"k": len(node_docs)})

        ranked_docs = retriever.invoke(question)
        path_rankings = [doc.metadata["path_idx"] for doc in ranked_docs]

        data_lines[nidx]["path_rankings"] = path_rankings
        data_lines[nidx]["html"] = str(soup)
        data_lines[nidx]["paths"] = paths
        data_lines[nidx]["is_leaf"] = is_leaf


    chat_tokenizer_path = "../../pretrained_model/meta-llama/Llama-3.2-3B-Instruct"
    chat_tokenizer = AutoTokenizer.from_pretrained(chat_tokenizer_path, trust_remote_code=True)

    def trim_html_tree_rerank(data_lines, context_window, dataset):
        for nidx in tqdm(range(len(data_lines)), desc=f"trim {dataset} {split} {context_window}"):
            paths = data_lines[nidx]['paths']
            html = data_lines[nidx]['html']
            is_leaf = data_lines[nidx]['is_leaf']
            path_rankings = data_lines[nidx]['path_rankings']

            max_context_window = int(context_window[:-1]) * 1000

            paths = [{"path": paths[i], "is_leaf": is_leaf[i]} for i in range(len(paths))]
            for idj in range(len(paths)):
                path_idx = int(path_rankings[idj])
                paths[path_idx]["ranking"] = idj

            soup = bs4.BeautifulSoup(html, 'html.parser')
            for idj in range(len(paths)):
                path = paths[idj]["path"]
                tag = soup
                for p in path:
                    for child in tag.contents:
                        if isinstance(child, bs4.element.Tag):
                            if child.name == p:
                                tag = child
                                break

                paths[idj]["tag"] = tag
                paths[idj]["token_length"] = len(chat_tokenizer.encode(str(tag), add_special_tokens=False))
            #  sort paths by ranking
            paths = sorted(paths, key=lambda x: x["ranking"])
            total_token_length = sum([p["token_length"] for p in paths])

            #  remove low ranking paths
            while total_token_length > max_context_window:
                if len(paths) == 1:
                    break
                discarded_path = paths.pop()
                total_token_length -= discarded_path["token_length"]
                trim_path(discarded_path)

            total_token_length = len(chat_tokenizer.encode(simplify_html(soup), add_special_tokens=False))
            while total_token_length > max_context_window:
                if len(paths) == 1:
                    break
                discarded_path = paths.pop()
                trim_path(discarded_path)
                total_token_length = len(chat_tokenizer.encode(simplify_html(soup), add_special_tokens=False))

            if total_token_length > max_context_window:
                # loguru.logger.warning(f"dataset {dataset} sample {idx} cannot be trimmed to {max_context_window} tokens")
                html_trim = truncate_input(simplify_html(soup), chat_tokenizer, max_context_window)
            else:
                html_trim = simplify_html(soup)

            assert len(chat_tokenizer.encode(
                html_trim,
                add_special_tokens=False)) <= max_context_window, f"html length: {len(chat_tokenizer.encode(html_trim, add_special_tokens=False))}, max_context_window: {max_context_window}"

            data_lines[nidx]["html_trim"] = html_trim
        return data_lines

    data_lines = trim_html_tree_rerank(data_lines, context_window, dataset)

    ckpt_path = "../../pretrained_model/zstanjj/HTML-Pruner-Llama-1B"
    max_context_window = re.match(r"(\d+)k", context_window).group(1)
    ratio = 2
    max_context_window = int(max_context_window) * 1000  // ratio
    #  remove low prob paths pointed tags
    loguru.logger.info(f"trimming htmls with context window {context_window}, max node words {max_node_words // 2}")
    # ckpt_path="/cpfs01/shared/public/guopeidong/models/glm4-9b/glm4-9b-128k-v0701-node2/checkpoint-1554"
    node_tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
    loguru.logger.info(f"node tokenizer: {node_tokenizer.name_or_path}, chat tokenizer: {chat_tokenizer.name_or_path}")

    thread_pool = []

    if torch.cuda.is_available():
        device = "cuda"
        parallel_size = torch.cuda.device_count()
        loguru.logger.info(f"Parallel size: {parallel_size}")
        shard_pool = []
    else:
        # model=AutoModelForCausalLM.from_pretrained("../../../huggingface/glm-4-9b-chat-1m",trust_remote_code=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(ckpt_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
        model.max_node_words = max_node_words // 2
        device = "cpu"
        model.to(device).eval()


    def init_shard_model(rank):
        shard_model = AutoModelForSeq2SeqLM.from_pretrained(ckpt_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
        shard_model.max_node_words = max_node_words // 2
        shard_model.to(f"cuda:{rank}").eval()
        shard_pool.append(shard_model)


    # #  copy model to all devices
    # if device == "cuda" and parallel_size > 1:
    #     for rank in range(parallel_size):
    #         thread = threading.Thread(target=init_shard_model, args=(rank,))
    #         thread.start()
    #         thread_pool.append(thread)
    #     for thread in thread_pool:
    #         thread.join()

    # total_len = len(data_lines)
    # res_lines = [{} for _ in range(total_len)]
    # pbar = tqdm(total=total_len, desc=f"Generation Pruning {dataset} {split}")

    output_dir = f"../../output/{dataset}/{model_name}/{_llm_name}"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/{model_name}-{dataset}-{max_node_words}-{context_window}.jsonl"
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)


    # def start_thread(rank):
    #     while len(data_lines) > 0:
    #         try:
    #             idx = total_len - len(data_lines)
    #             data_line = data_lines.pop(0)
    #             question = data_line['question']
    #             coarse_html_trim = data_line["html_trim"]
    #             html_res = shard_pool[rank].generate_html_tree(node_tokenizer, [question], [coarse_html_trim])
    #             # loguru.logger.info(f"Calculate probs for {len(html_res[0]['path_probs'])} nodes")
    #             data_line.pop(f'search_results', None)
    #
    #             res_lines[idx] = {**data_line, **html_res[0]}
    #             res_lines[idx]["html_trim"] = trim_html_tree(
    #                 html=html_res[0]["html"],
    #                 paths=html_res[0]["paths"],
    #                 is_leaf=html_res[0]["is_leaf"],
    #                 node_tree=html_res[0]["node_tree"],
    #                 chat_tokenizer=chat_tokenizer,
    #                 node_tokenizer=node_tokenizer,
    #                 max_context_window=max_context_window,
    #             )
    #             res_lines[idx]["coarse_html_trim"] = coarse_html_trim
    #
    #
    #         except Exception as e:
    #             loguru.logger.error(f"Error in processing line {idx}: {e}")
    #             traceback.print_exc()
    #             # print(f"Error in processing line {idx}: {e}")
    #             #  save the processed data
    #             with open(output_file, "w") as f:
    #                 for idx in range(len(res_lines)):
    #                     #  convert "path_probs" from float32 to string
    #                     # res_lines[idx]["path_probs"] = [str(prob) for prob in res_lines[idx]["path_probs"]]
    #                     try:
    #                         f.write(json.dumps(res_lines[idx], ensure_ascii=False) + "\n")
    #                     except Exception as e:
    #                         # loguru.logger.error(f"Error in writing line {idx}: {e}")
    #                         f.write(json.dumps(res_lines[idx], ensure_ascii=True) + "\n")
    #         pbar.update(1)
    #
    #
    # for i in range(len(shard_pool)):
    #     thread = threading.Thread(target=start_thread, args=(i,))
    #     thread.start()
    #     thread_pool.append(thread)
    #
    # for thread in thread_pool:
    #     thread.join()
    #
    # pbar.close()

    data_dict = {}
    for l in data_lines:
        data_dict[l["id"]] = {"question": l["question"],
                              "query_time": l["query_time"],
                              # "coarse_html_trim": l["coarse_html_trim"],
                              "html_trim": l["html_trim"],
                              }
    with open(output_file, 'w') as json_file:
        json.dump(data_dict, json_file, indent=4)
    loguru.logger.info(f"Saved parsed html to {output_file}")
