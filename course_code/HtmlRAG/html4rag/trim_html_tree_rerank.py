import argparse

import loguru
from bs4 import BeautifulSoup
from tqdm import tqdm
from transformers import AutoTokenizer
import sys
sys.path.append("./")
from html4rag.html_utils import *
from multiprocessing import Process
from html4rag.html_utils import simplify_html, truncate_input, trim_path
import bs4
import json

query_instruction_for_retrieval = "Represent this sentence for searching relevant passages: "


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--split", type=str, default="test")
    argparser.add_argument("--search_engine", type=str, default="bing")
    argparser.add_argument("--rewrite_method", type=str, default="slimplmqr")
    argparser.add_argument("--rerank_model", type=str, default="bgelargeen")
    argparser.add_argument("--mini_dataset", action="store_true")
    argparser.add_argument("--chat_tokenizer_name", type=str, default="llama")
    argparser.add_argument("--max_node_words", type=int, default=256)
    args = argparser.parse_args()
    split = args.split
    search_engine = args.search_engine
    rewrite_method = args.rewrite_method
    rerank_model = args.rerank_model
    max_node_words = args.max_node_words

    chat_tokenizer_name = args.chat_tokenizer_name


    chat_tokenizer_path = "../../huggingface/Meta-Llama-3.1-8B-Instruct"
    chat_tokenizer = AutoTokenizer.from_pretrained(chat_tokenizer_path, trust_remote_code=True)

    def trim_html_tree_rerank(context_window, dataset):
        data_file = f"./html_data/{dataset}/tree-rerank/{search_engine}html-{rewrite_method}-{rerank_model}-{max_node_words}-{dataset}-{split}.jsonl"
        data_lines = [json.loads(line) for line in open(data_file)]
        loguru.logger.info(f"read {len(data_lines)} node lines from {data_file}")
        output_file = f"./html_data/{dataset}/tree-rerank/{chat_tokenizer_name}/{search_engine}html-{rewrite_method}-{rerank_model}-{max_node_words}-{dataset}-{split}-{context_window}.jsonl"
        if args.mini_dataset:
            data_lines = data_lines[:10]

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
        with open(output_file, "w") as f:
            for idx in range(len(data_lines)):
                f.write(json.dumps(data_lines[idx], ensure_ascii=False) + "\n")
        loguru.logger.info(f"write {len(data_lines)} node lines to {output_file}")

    processes = []
    # context_windows = ["2k", "4k", "6k", "8k", "10k", "12k", "16k", "20k", "32k", "40k", "64k"]
    context_windows = ["2k", "4k", "6k", "8k", "12k", "16k", "32k"]
    # context_windows = ["2k"]
    datasets = ["asqa", "hotpot-qa", "nq", "trivia-qa", "musique"]
    # datasets = ["eli5"]

    for context_window in context_windows:
        for dataset in datasets:
            p = Process(target=trim_html_tree_rerank, args=(context_window, dataset))
            p.start()
            processes.append(p)

        if len(processes) >= 8:
            for p in processes:
                p.join()

    if len(processes) > 0:
        for p in processes:
            p.join()

    loguru.logger.info("all processes finished")

