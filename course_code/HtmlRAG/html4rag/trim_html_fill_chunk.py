import argparse
import json
import multiprocessing
import os
import re

import loguru
from tqdm import tqdm
from transformers import AutoTokenizer
import sys
sys.path.append("./")
from html4rag.html_utils import truncate_input


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--split", type=str, default="test")
    argparser.add_argument("--mini_dataset", action="store_true")
    argparser.add_argument("--rewrite_method", type=str, default="slimplmqr")
    argparser.add_argument("--rerank_model", type=str, default="bgelargeen")
    argparser.add_argument("--chat_tokenizer_name", type=str, default="llama")
    args = argparser.parse_args()

    split = args.split
    rewrite_method = args.rewrite_method
    rerank_model = args.rerank_model
    search_engine = "bing"
    chat_tokenizer_name = args.chat_tokenizer_name

    if chat_tokenizer_name == "bc":
        chat_tokenizer_path = "../../huggingface/Baichuan2-7B-Chat/"
    elif chat_tokenizer_name == "llama":
        chat_tokenizer_path = "../../huggingface/Meta-Llama-3.1-70B-Instruct/"
    else:
        raise ValueError(f"unknown tokenizer {chat_tokenizer_name}")
    chat_tokenizer = AutoTokenizer.from_pretrained(chat_tokenizer_path, trust_remote_code=True)

    # context_windows = ["192k", "128k", "64k", "32k", "16k", "8k", "4k", "2k"]
    context_windows = ["32k", "16k", "8k", "4k", "2k"]
    datasets = ["asqa", "hotpot-qa", "nq", "trivia-qa", "musique", "eli5"]
    # context_windows = ["8k"]
    # datasets = ["eli5"]

    def fill_chunk(context_window, dataset):
        input_file = f"./html_data/{dataset}/{search_engine}/{search_engine}html-{rewrite_method}-{rerank_model}-{dataset}-simple-{split}.jsonl"
        output_dir = f"./html_data/{dataset}/baselines/{chat_tokenizer_name}"
        output_file = f"{output_dir}/{search_engine}html-{rewrite_method}-{rerank_model}-{dataset}-{split}-{context_window}.jsonl"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        data_lines = [json.loads(l) for l in open(input_file)]
        loguru.logger.info(f"loaded {len(data_lines)} lines from {input_file}")
        # html_simple_lines = [json.loads(l) for l in open(html_simple_file)]
        if args.mini_dataset:
            data_lines = data_lines[:10]

        for idx in tqdm(range(len(data_lines)), desc=f"{dataset} trimming htmls with context window {context_window}"):
            #  fill in chunks until the length of the input exceeds the max length
            chunks = data_lines[idx]['page_contents']
            max_context_window = re.match(r"(\d+)k", context_window).group(1)
            max_context_window = int(max_context_window) * 1000

            ref_chunks = chunks[:1]
            ref_token_length = len(chat_tokenizer.encode(" ".join(ref_chunks), add_special_tokens=False))
            for i in range(1, len(chunks)):
                if ref_token_length > max_context_window:
                    break
                ref_chunks.append(chunks[i])
                ref_token_length += len(chat_tokenizer.encode(chunks[i], add_special_tokens=False))
            if len(ref_chunks) == 1:
                ref_chunks[0] = truncate_input(ref_chunks[0], chat_tokenizer, max_context_window)
            else:
                while True:
                    ref_chunks = ref_chunks[:-1]
                    total_token_length = len(chat_tokenizer.encode(" ".join(ref_chunks), add_special_tokens=False))
                    if total_token_length <= max_context_window:
                        break
            total_token_length = len(chat_tokenizer.encode(" ".join(ref_chunks), add_special_tokens=False))
            assert total_token_length <= max_context_window, f"total token length {total_token_length} exceeds {max_context_window}"

            data_lines[idx]['html_trim'] = ref_chunks

        with open(output_file, "w") as f:
            for l in data_lines:
                f.write(json.dumps(l, ensure_ascii=False) + "\n")
        loguru.logger.info(
            f"html trimmed with context window {context_window}")
        loguru.logger.info(f"saved to {output_file}")

    max_processes = 8
    process_pool=[]
    for context_window in context_windows:
        for dataset in datasets:
            p = multiprocessing.Process(target=fill_chunk, args=(context_window, dataset))
            p.start()
            process_pool.append(p)

        if len(process_pool) >= max_processes:
            for process in process_pool:
                process.join()
            process_pool = []

    if len(process_pool) > 0:
        for process in process_pool:
            process.join()

    loguru.logger.info("All processes finished")

