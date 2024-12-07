
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

dataset_path = "../example_data/dev_data.jsonl.bz2"
for batch in tqdm(load_data_in_batches(dataset_path, 1, -1), desc="Generating predictions"):
    batch_ground_truths = batch.pop("answer")  # Remove answers from batch and store them
    print(batch_ground_truths)
