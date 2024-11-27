import bz2
import json
import os
from datetime import datetime
import argparse
import requests
from loguru import logger
from tqdm.auto import tqdm
from bs4 import BeautifulSoup
from html4rag.html_utils import simplify_html

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


def generate_predictions(dataset_path, model, split):
    """
    Processes batches of data from a dataset to generate predictions using a model.

    Args:
    dataset_path (str): Path to the dataset.
    model (object): UserModel that provides `get_batch_size()` and `batch_generate_answer()` interfaces.

    Returns:
    tuple: A tuple containing lists of queries, ground truths, and predictions.
    """
    interaction_id, queries, answers, search_results = [], [], [], []
    # batch_size = model.get_batch_size()

    for batch in tqdm(load_data_in_batches(dataset_path, 1, split), desc="Generating Simplify HTML"):
        # batch_ground_truths = batch.pop("answer")  # Remove answers from batch and store them
        interaction_id.extend(batch["interaction_id"])
        queries.extend(batch["query"])
        answers.extend(batch["answer"])
        cur_search_results = []
        for search in batch["search_results"][0]:
            html_source = search["page_result"]
            h_soup = BeautifulSoup(html_source, 'html.parser')
            html_source = simplify_html(h_soup, keep_attr=True)
            cur_search_results.append(html_source)
        search_results.append(cur_search_results)
    return interaction_id, queries, answers, search_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path", type=str, default="../example_data/dev_data.jsonl.bz2",
                        choices=["../example_data/dev_data.jsonl.bz2",  # example data
                                 "../data/crag_task_1_dev_v4_release.jsonl.bz2",  # full data
                                 ])
    parser.add_argument("--split", type=int, default=1,
                        help="The split of the dataset to use. This is only relevant for the full data: "
                             "0 for public validation set, 1 for public test set")

    parser.add_argument("--model_name", type=str, default="htmlrag",
                        choices=["vanilla_baseline",
                                 "rag_baseline",
                                 "multifeature",
                                 "htmlrag"
                                 # add your model here
                                 ],
                        )

    parser.add_argument("--llm_name", type=str, default="../pretrained_model/meta-llama/Llama-3.2-3B-Instruct",
                        choices=["../pretrained_model/meta-llama/Llama-3.2-3B-Instruct",
                                 "../pretrained_model/google/gemma-2-2b-it",
                                 # can add more llm models here
                                 ])
    parser.add_argument("--is_server", action="store_true", default=True,
                        help="Whether we use vLLM deployed on a server or offline inference.")
    parser.add_argument("--vllm_server", type=str, default="http://localhost:8088/v1",
                        help="URL of the vLLM server if is_server is True. The port number may vary.")

    args = parser.parse_args()
    print("Is Server?", args.is_server)

    dataset_path = args.dataset_path
    dataset = dataset_path.split("/")[0]
    split = -1
    if dataset == "data":
        split = args.split
        if split == -1:
            raise ValueError("Please provide a valid split value for the full data: "
                             "0 for public validation set, 1 for public test set.")
    dataset_path = os.path.join("..", dataset_path)

    llm_name = args.llm_name
    _llm_name = llm_name.split("/")[-1]

    model_name = args.model_name
    model = None

    # Generate predictions
    interaction_id, queries, answers, search_results = generate_predictions(dataset_path, model, split)

    # Combine results into a JSON-serializable structure
    data = [
        {
            "id": interaction_id[i],
            "question": queries[i],
            "answers": [answers[i]],
            "search_results": search_results[i]
        }
        for i in range(len(interaction_id))
    ]

    # Dump to a JSON file
    with open("./html_data/kdd_crag/kdd_crag.jsonl", "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

    print("Results saved to ./html_data/kdd_crag/kdd_crag.jsonl")


