import bz2
import json
import os
from datetime import datetime
import argparse

from loguru import logger
from tqdm.auto import tqdm
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
    queries, ground_truths, predictions = [], [], []
    batch_size = model.get_batch_size()

    for batch in tqdm(load_data_in_batches(dataset_path, batch_size, split), desc="Generating predictions"):
        batch_ground_truths = batch.pop("answer")  # Remove answers from batch and store them
        batch_predictions = model.batch_generate_answer(batch)
        queries.extend(batch["query"])
        ground_truths.extend(batch_ground_truths)
        predictions.extend(batch_predictions)

    return queries, ground_truths, predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path", type=str, default="data/crag_task_1_dev_v4_release.jsonl.bz2",
                        choices=["example_data/dev_data.jsonl.bz2", # example data
                                 "data/crag_task_1_dev_v4_release.jsonl.bz2", # full data
                                 ])
    parser.add_argument("--split", type=int, default=1,
                        help="The split of the dataset to use. This is only relevant for the full data: "
                             "0 for public validation set, 1 for public test set")

    parser.add_argument("--model_name", type=str, default="db3",
                        choices=["vanilla_baseline",
                                 "rag_baseline",
                                 "multifeature",
                                 "htmlrag",
                                 "bge_baseline",
                                 "db3"
                                 # add your model here
                                 ],
                        )

    parser.add_argument("--llm_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct",
                        choices=["../pretrained_model/meta-llama/Llama-3.2-3B-Instruct",
                                 "../pretrained_model/google/gemma-2-2b-it",
                                 "meta-llama/Meta-Llama-3-8B-Instruct",

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
    if "Llama-3-8B-Instruct" in llm_name:
        _llm_name = "Llama-3-8B-Instruct"
    else:
        _llm_name = llm_name.split("/")[-1]
    output_file = f"predictions.json"
    model_name = args.model_name
    if model_name == "vanilla_baseline":
        from vanilla_baseline import InstructModel
        model = InstructModel(llm_name=llm_name, is_server=args.is_server, vllm_server=args.vllm_server)
    elif model_name == "rag_baseline":
        from rag_baseline import RAGModel
        model = RAGModel(llm_name=llm_name, is_server=args.is_server, vllm_server=args.vllm_server)
    elif model_name == "multifeature":
        from rag_multifeature_baseline import RAGModel
        model = RAGModel(llm_name=llm_name, is_server=args.is_server, vllm_server=args.vllm_server, device="cuda:1")
    elif model_name == "htmlrag":
        from rag_htmlrag_baseline import RAGModel
        output_directory = os.path.join("..", "output", dataset, model_name, _llm_name)
        max_node_words = 1000
        context_window = "4k"
        input_file = f"{output_directory}/{model_name}-{dataset}-{max_node_words}-{context_window}.jsonl"
        output_file = f"{max_node_words}-{context_window}-predictions.json"
        model = RAGModel(llm_name=llm_name, is_server=args.is_server, vllm_server=args.vllm_server, input_file=input_file)
    elif model_name == "bge_baseline":
        from bge_baseline import RAGModel
        output_directory = os.path.join("..", "output", dataset, model_name, _llm_name)
        rerank_model = "bgelargeen"
        input_file = f"{output_directory}/{rerank_model}-{dataset}.jsonl"
        model = RAGModel(llm_name=llm_name, is_server=args.is_server, vllm_server=args.vllm_server, input_file=input_file)
    elif model_name == "db3":
        from rag_db3_baseline import RAGModel
        model = RAGModel(llm_name=llm_name)
    else:
        raise ValueError("Model name not recognized.")

    # make output directory
    output_directory = os.path.join("..", "output", dataset, model_name, _llm_name)
    os.makedirs(output_directory, exist_ok=True)

    # Generate predictions
    queries, ground_truths, predictions = generate_predictions(dataset_path, model, split)

    # save predictions
    json.dump({"queries": queries, "ground_truths": ground_truths, "predictions": predictions},
              open(os.path.join(output_directory, output_file), "w"), indent=4)
