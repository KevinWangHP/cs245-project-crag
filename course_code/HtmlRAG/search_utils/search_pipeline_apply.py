import argparse
import json
import sys
import os
from time import sleep

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from tqdm import tqdm
from search_utils.bing_search import BingSearch


def stable_search(query):
    patience = 10
    search_results = []
    while True:
        try:
            search_results = BingSearch.search(query)
            break
        except Exception as e:
            print(f"search query {query} failed, error: {e}")
            patience -= 1
            sleep(1)
            if patience <= 0:
                break
    return search_results


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--dataset", type=str, default="asqa", help="dataset name")
    arg_parser.add_argument("--split", type=str, default="test", help="split name")
    arg_parser.add_argument("--search_method", type=str, default="vanilla_search", help="search method")
    arg_parser.add_argument("--mini_dataset", action="store_true", help="whether to use mini dataset")
    args = arg_parser.parse_args()
    dataset = args.dataset
    split = args.split
    search_method = args.search_method

    if search_method == "vanilla_search":
        input_file = f"./html_data/{dataset}/{dataset}-{split}.jsonl"
        output_dir = f"./html_data/{dataset}/bing"
        output_file = f"{output_dir}/bing-{dataset}-{split}.jsonl"
    elif search_method in ["slimplmqr", ]:
        input_file = f"./html_data/{dataset}/{search_method}/{search_method}-{dataset}-{split}.jsonl"
        output_dir = f"./html_data/{dataset}/bing"
        output_file = f"{output_dir}/bing-{search_method}-{dataset}-{split}.jsonl"
    else:
        raise NotImplementedError(f"search method {search_method} not implemented")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(input_file, "r", encoding="utf-8") as f:
        data_lines = [json.loads(line) for line in f]

    if args.mini_dataset:
        data_lines = data_lines[:10]
    print(f"bing search dataset {dataset}, split {split}, search method {search_method}")

    with open(output_file, "w", encoding="utf-8") as f:
        pass

    for idx, data_line in tqdm(enumerate(data_lines), total=len(data_lines), desc=f"bing-{dataset}-{split}"):
        #  do search with vanilla query
        if search_method == "vanilla_search":
            query = data_line["question"]
            if idx == 0:
                print(query)
            try:
                vanilla_query_results = stable_search(query)
            except:
                print(f"error: {idx}, {query}")
                vanilla_query_results = []
            data_lines[idx][f"{search_method}_results"] = vanilla_query_results

        elif search_method in ["slimplmqr", ]:
            queries = [data_line["question"]]
            queries += [q["question"] for q in data_line[f"{search_method}_rewrite"]["questions"] if q["question"]]
            #  restrict to 5 queries
            queries = queries[:5]
            if idx == 0:
                print(queries)

            query_results = []
            for query in queries:
                try:
                    query_results.extend(stable_search(query))
                except:
                    print(f"error: {idx}, {query}")
            data_lines[idx][f"{search_method}_results"] = query_results

        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(data_lines[idx], ensure_ascii=False) + "\n")
