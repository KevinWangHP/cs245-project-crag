from htmlrag import clean_html
from htmlrag import build_block_tree
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
from HtmlRAG.html4rag.html_utils import trim_path, simplify_html
import numpy as np


def prune_HTML_top300(self, html, block_tree: List[Tuple], block_rankings: List[int], chat_tokenizer):
    """
    根据 block_rankings 的排名，保留排名前300的块，其余全部丢弃。

    Args:
        html (str): 原始HTML文档。
        block_tree (List[Tuple]): 块树结构，包含每个块的路径信息。
        block_rankings (List[int]): 每个块的排名。
        chat_tokenizer: 用于HTML编码的分词器。

    Returns:
        str: 修剪后的HTML文档。
    """
    from bs4 import BeautifulSoup

    # 提取路径和叶子节点信息
    paths = [b[1] for b in block_tree]
    is_leaf = [b[2] for b in block_tree]

    # 构建包含路径、叶子节点标志和排名的字典
    paths = [{"path": paths[i], "is_leaf": is_leaf[i], "ranking": block_rankings[i]} for i in range(len(paths))]

    # 按排名排序，取前300名
    top300_paths = sorted(paths, key=lambda x: x["ranking"])[:300]

    # 解析HTML
    soup = BeautifulSoup(html, 'html.parser')

    # 标记保留的块
    for path in top300_paths:
        tag = soup
        for p in path["path"]:  # 根据路径找到对应的标签
            for child in tag.contents:
                if isinstance(child, bs4.element.Tag) and child.name == p:
                    tag = child
                    break
        # 标记需要保留的tag
        path["tag"] = tag

    # 移除不在前300名的块
    def remove_unwanted_tags(tag):
        """
        递归移除HTML文档中不在前300路径中的内容。
        """
        for child in tag.contents[:]:
            if isinstance(child, bs4.element.Tag) and child not in [p["tag"] for p in top300_paths]:
                child.decompose()  # 删除不需要的标签
            elif isinstance(child, bs4.element.Tag):
                remove_unwanted_tags(child)  # 递归处理子标签

    remove_unwanted_tags(soup)

    # 简化和返回HTML
    html_trim = simplify_html(soup)
    return html_trim




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

def query_model(question, html):
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
    # print(html)
    simplified_html = clean_html(html)#for
    # print(simplified_html)

    # <html>
    # <title>When was the bellagio in las vegas built?</title>
    # <p>The Bellagio is a luxury hotel and casino located on the Las Vegas Strip in Paradise, Nevada. It was built in 1998.</p>
    # <div>
    # <p>Some other text</p>
    # <p>Some other text</p>
    # </div>
    # </html>

    block_tree, simplified_html = build_block_tree(simplified_html, max_node_words=10)
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
    #######
    block_rankings = embed_html_pruner.calculate_block_rankings(question, simplified_html, block_tree)#for
    ########
    # print(block_rankings)
    print('**********************************************')
    # min_indices = np.argsort(block_rankings)[:2]
    # print(min_indices)
    # for i in range(len(min_indices)):
    #     print(block_rankings[min_indices[i]])
    # min_indices = list(min_indices)
    # print(block_rankings[int(min_indices)])
    # [0, 2, 1]

    chat_tokenizer = AutoTokenizer.from_pretrained("../pretrained_model/meta-llama/Llama-3.1-70B-Instruct")

    max_context_window = 512
    pruned_html = prune_HTML_top300(simplified_html, block_tree, block_rankings, chat_tokenizer)
    # pruned_html = embed_html_pruner.prune_HTML(simplified_html, block_tree, block_rankings, chat_tokenizer, max_context_window)
    # print(pruned_html)
    # <html>
    # <title>When was the bellagio in las vegas built?</title>
    # <p>The Bellagio is a luxury hotel and casino located on the Las Vegas Strip in Paradise, Nevada. It was built in 1998.</p>
    # </html>
    print("*****stop*****")
    # ckpt_path = "zstanjj/HTML-Pruner-Llama-1B"
    # if torch.cuda.is_available():
    #     device = "cuda:1"
    # else:
    #     device = "cpu"
    # gen_embed_pruner = GenHTMLPruner(gen_model=ckpt_path, max_node_words=5, device=device)
    # soup = bs4.BeautifulSoup("", 'html.parser')
    # soup.append(bs4.BeautifulSoup(pruned_html, 'html.parser'))
    # res = split_tree(soup, 10)
    # block_rankings = gen_embed_pruner.calculate_block_rankings(question, pruned_html)
    # print(block_rankings)
    # [1, 0]

    # block_tree, pruned_html = build_block_tree(pruned_html, max_node_words=10)
    # print(len(block_rankings))
    # print(len(block_tree))
    # print(len(pruned_html))

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

    max_context_window = 256
    #pruned_html = gen_embed_pruner.prune_HTML(pruned_html, block_tree, block_rankings, chat_tokenizer, max_context_window)
    # skip prune

    # print(pruned_html)
    return pruned_html

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
        # html_source = batch["search_results"][0][3]["page_result"]#.0-4拼一起
        html_source = "".join([batch["search_results"][0][i]["page_result"] for i in range(5)])
        # html_source = query_model(batch["query"][0], html_source)
        search_results = query_model(batch["query"][0], html_source)
    return interaction_id, queries, answers, search_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path", type=str, default="example_data/dev_data.jsonl.bz2",
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
