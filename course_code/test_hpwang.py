from htmlrag import EmbedHTMLPruner
from htmlrag import GenHTMLPruner
import torch
from transformers import AutoTokenizer
import bz2
import json
import os
from datetime import datetime
import argparse
import requests
from loguru import logger
from tqdm.auto import tqdm
from bs4 import BeautifulSoup
from typing import List, Tuple
from collections import defaultdict
import bs4
from HtmlRAG.html4rag.html_utils import *
import numpy as np
from htmlrag import clean_html, build_block_tree


def split_tree(soup: bs4.BeautifulSoup, max_node_words=0) -> List[Tuple[bs4.element.Tag, List[str], bool]]:
    word_count = len(soup.get_text().split())
    if word_count > max_node_words:
        possible_trees = [(soup, [])]
        target_trees = []  # [(tag, path, is_leaf)]
        #  split the entire dom tee into subtrees, until the length of the subtree is less than max_node_words words
        #  find all possible trees
        while True:
            if len(possible_trees) == 0:
                break
            tree = possible_trees.pop(0)
            tag_children = defaultdict(int)
            bare_word_count = 0
            #  count child tags
            for child in tree[0].contents:
                if isinstance(child, bs4.element.Tag):
                    tag_children[child.name] += 1
            _tag_children = {k: 0 for k in tag_children.keys()}

            #  check if the tree can be split
            for child in tree[0].contents:
                if isinstance(child, bs4.element.Tag):
                    #  change child tag with duplicate names
                    if tag_children[child.name] > 1:
                        new_name = f"{child.name}{_tag_children[child.name]}"
                        new_tree = (child, tree[1] + [new_name])
                        _tag_children[child.name] += 1
                        child.name = new_name
                    else:
                        new_tree = (child, tree[1] + [child.name])
                    word_count = len(child.get_text().split())
                    #  add node with more than max_node_words words, and recursion depth is less than 64
                    if word_count > max_node_words and len(new_tree[1]) < 64:
                        possible_trees.append(new_tree)
                    else:
                        target_trees.append((new_tree[0], new_tree[1], True))
                else:
                    bare_word_count += len(str(child).split())

            #  add leaf node
            if len(tag_children) == 0:
                target_trees.append((tree[0], tree[1], True))
            #  add node with more than max_node_words bare words
            elif bare_word_count > max_node_words:
                target_trees.append((tree[0], tree[1], False))
    else:
        soup_children = [c for c in soup.contents if isinstance(c, bs4.element.Tag)]
        if len(soup_children) == 1:
            target_trees = [(soup_children[0], [soup_children[0].name], True)]
        else:
            # add an html tag to wrap all children
            new_soup = bs4.BeautifulSoup("", 'html.parser')
            new_tag = new_soup.new_tag("html")
            new_soup.append(new_tag)
            for child in soup_children:
                new_tag.append(child)
            target_trees = [(new_tag, ["html"], True)]
    return target_trees

def query_model(question, html, max_context_window_1, max_context_window_2):
    # question = "When was the bellagio in las vegas built?"
    # html = """
    # <html>
    # <head>
    # <title>When was the bellagio in las vegas built?</title>
    # </head>
    # <body>
    # <p class="class0">The Bellagio is a luxury hotel and casino located on the Las Vegas Strip in Paradise, Nevada. It was built in 1998.</p>
    # </body>
    # <div>
    # <div>
    # <p>Some other text</p>
    # <p>Some other text</p>
    # </div>
    # </div>
    # <p class="class1"></p>
    # <!-- Some comment -->
    # <script type="text/javascript">
    # document.write("Hello World!");
    # </script>
    # </html>
    # """
    print(f"\nHTML Length Before Cleaning: {len(html)}")
    simplified_html = clean_html(html)#for
    print(f"HTML Length After Cleaning: {len(simplified_html)}")

    # <html>
    # <title>When was the bellagio in las vegas built?</title>
    # <p>The Bellagio is a luxury hotel and casino located on the Las Vegas Strip in Paradise, Nevada. It was built in 1998.</p>
    # <div>
    # <p>Some other text</p>
    # <p>Some other text</p>
    # </div>
    # </html>

    block_tree, simplified_html = build_block_tree(simplified_html, max_node_words=20)
    # print(f"Block Tree Length After Cleaning: {len(block_tree)}")
    # for block in block_tree:
    #     print("Block Content: ", block[0])
    #     print("Block Path: ", block[1])
    #     print("Is Leaf: ", block[2])
    #     print("")

    # Block Content:  <title>When was the bellagio in las vegas built?</title>
    # Block Path:  ['html', 'title']
    # Is Leaf:  True
    #
    # Block Content:  <div>
    # <p>Some other text</p>
    # <p>Some other text</p>
    # </div>
    # Block Path:  ['html', 'div']
    # Is Leaf:  True
    #
    # Block Content:  <p>The Bellagio is a luxury hotel and casino located on the Las Vegas Strip in Paradise, Nevada. It was built in 1998.</p>
    # Block Path:  ['html', 'p']
    # Is Leaf:  True

    embed_html_pruner = EmbedHTMLPruner(embed_model="../pretrained_model/BAAI/bge-large-en")

    block_rankings = embed_html_pruner.calculate_block_rankings(question, simplified_html, block_tree)#for

    chat_tokenizer = AutoTokenizer.from_pretrained("../pretrained_model/meta-llama/Llama-3.1-70B-Instruct")

    max_context_window = max_context_window_1
    # pruned_html = prune_HTML_top300(simplified_html, block_tree, block_rankings, chat_tokenizer)
    pruned_html_res = embed_html_pruner.prune_HTML(simplified_html, block_tree, block_rankings, chat_tokenizer, max_context_window)
    print(f"Pruned HTML Length: {len(pruned_html_res)}")
    # print(pruned_html)
    # <html>
    # <title>When was the bellagio in las vegas built?</title>
    # <p>The Bellagio is a luxury hotel and casino located on the Las Vegas Strip in Paradise, Nevada. It was built in 1998.</p>
    # </html>
    ckpt_path = "zstanjj/HTML-Pruner-Llama-1B"
    if torch.cuda.is_available():
        device = "cuda:1"
    else:
        device = "cpu"
    gen_embed_pruner = GenHTMLPruner(gen_model=ckpt_path, max_node_words=10, device=device)
    block_rankings = gen_embed_pruner.calculate_block_rankings(question, pruned_html_res)
    # print(f"Block Ranking Length After Pruning: {len(block_rankings)}")
    # # [1, 0]
    #
    block_tree, pruned_html_res = build_block_tree(pruned_html_res, max_node_words=10)
    # print(f"Block Tree Length After Pruning: {len(block_tree)}")


    # for block in block_tree:
    #     print("Block Content: ", block[0])
    #     print("Block Path: ", block[1])
    #     print("Is Leaf: ", block[2])
    #     print("")

    # Block Content:  <title>When was the bellagio in las vegas built?</title>
    # Block Path:  ['html', 'title']
    # Is Leaf:  True
    #
    # Block Content:  <p>The Bellagio is a luxury hotel and casino located on the Las Vegas Strip in Paradise, Nevada. It was built in 1998.</p>
    # Block Path:  ['html', 'p']
    # Is Leaf:  True

    # max_context_window = max_context_window_2
    # gen_pruned_html_res = gen_embed_pruner.prune_HTML(pruned_html_res, block_tree, block_rankings, chat_tokenizer, max_context_window)
    # print(f"HTML Length After Gen Pruning: {len(gen_pruned_html_res)}")
    gen_pruned_html_res = None

    # skip prune

    # print(pruned_html)
    return pruned_html_res, gen_pruned_html_res

    # <p>The Bellagio is a luxury hotel and casino located on the Las Vegas Strip in Paradise, Nevada. It was built in 1998.</p>






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


def generate_predictions(dataset_path, model, split, max_context_window_1, max_context_window_2):
    """
    Processes batches of data from a dataset to generate predictions using a model.

    Args:
    dataset_path (str): Path to the dataset.
    model (object): UserModel that provides `get_batch_size()` and `batch_generate_answer()` interfaces.

    Returns:
    tuple: A tuple containing lists of queries, ground truths, and predictions.
    """
    interaction_id, queries, answers, pruned_html, gen_pruned_html = [], [], [], [], []
    # batch_size = model.get_batch_size()
    i = 0
    for batch in tqdm(load_data_in_batches(dataset_path, 1, split), desc="Generating Retrieval Result"):
        # batch_ground_truths = batch.pop("answer")  # Remove answers from batch and store them
        interaction_id.extend(batch["interaction_id"])
        queries.extend(batch["query"])
        answers.extend(batch["answer"])
        cur_search_results = []
        # html_source = batch["search_results"][0][3]["page_result"]#
        html_source = "\n".join([batch["search_results"][0][i]["page_result"] for i in range(5)])
        # html_source = query_model(batch["query"][0], html_source)
        pruned, gen_pruned = query_model(batch["query"][0], html_source, max_context_window_1, max_context_window_2)
        pruned_html.append(pruned)
        gen_pruned_html.append(gen_pruned)
    return interaction_id, queries, answers, pruned_html, gen_pruned_html


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path", type=str, default="data/crag_task_1_dev_v4_release.jsonl.bz2",
                        choices=["example_data/dev_data.jsonl.bz2",  # example data
                                 "data/crag_task_1_dev_v4_release.jsonl.bz2",  # full data
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

    max_context_window_1 = 1000
    max_context_window_2 = 1000

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
    interaction_id, queries, answers, pruned_html, gen_pruned_html = generate_predictions(dataset_path, model, split, max_context_window_1, max_context_window_2)

    file_path = f"../output/preprocess/htmlrag/{dataset}/"
    file_name_pruned = f"pruned_html_{max_context_window_1}.json"
    # file_name_gen_pruned = "gen_pruned_html.json"
    os.makedirs(file_path, exist_ok=True)

    data_to_dump = dict(zip(interaction_id, pruned_html))
    # Write the dictionary to the JSON file
    with open(os.path.join(file_path, file_name_pruned), "w") as file:
        json.dump(data_to_dump, file, indent=4)

    # data_to_dump = dict(zip(interaction_id, gen_pruned_html))
    # # Write the dictionary to the JSON file
    # with open(os.path.join(file_path, file_name_gen_pruned), "w") as file:
    #     json.dump(data_to_dump, file, indent=4)

