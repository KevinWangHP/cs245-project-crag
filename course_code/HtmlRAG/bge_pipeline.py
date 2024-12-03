import bz2
import argparse
import bz2
import concurrent.futures
import json
import multiprocessing
import os
import re
import sys

import loguru
from bs4 import BeautifulSoup
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import HTMLHeaderTextSplitter, RecursiveCharacterTextSplitter
from loguru import logger
from tqdm import tqdm
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from html4rag.html_utils import simplify_html

sys.path.append("./")
from html4rag.html_utils import truncate_input
from html4rag.html_utils import HTMLSplitter, headers_to_split_on, clean_xml



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

    parser.add_argument("--dataset_path", type=str, default="../example_data/dev_data.jsonl.bz2",
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
    parser.add_argument("--model_name", type=str, default="bm25",
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
    parser.add_argument("--context_window", type=str, default="32k")
    parser.add_argument("--chunk_size", type=int, default=1000)
    args = parser.parse_args()
    context_window = args.context_window
    model_name = args.model_name
    url = args.url
    dataset_path = args.dataset_path
    dataset = dataset_path.split("/")[1]
    llm_name = args.llm_name
    _llm_name = llm_name.split("/")[-1]
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



    tokenizer_path = "../../pretrained_model/BAAI/bge-large-en"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if model_name == "bge_baseline":
        embedder = HuggingFaceBgeEmbeddings(
            model_name=url,
            model_kwargs={'device': args.device},
            encode_kwargs={'normalize_embeddings': True},
        )
        query_instruction_for_retrieval = "Represent this sentence for searching relevant passages: "
    elif model_name == "e5-mistral":
        query_instruction_for_retrieval = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
    elif model_name == "bm25":
        embedder = None
        query_instruction_for_retrieval = ""
    else:
        raise NotImplementedError(f"rerank model {model_name} not implemented")


    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for idx, data_line in tqdm(enumerate(data_lines), total=len(data_lines), desc="Reranking"):
            question = query_instruction_for_retrieval + data_line["question"] + "query time:" + data_line["query_time"]
            docs = []
            html = [clean_xml(d) for d in data_lines[idx][f'search_results']]
            html = [h for h in html if h.strip()]
            html_indexes = []
            for idh, h in enumerate(html):
                splits = html_splitter.split_text(h)
                splits = text_splitter.split_documents(splits)
                html_indexes.extend([idh] * len(splits))
                docs.extend(splits)
            # db = FAISS.from_documents(docs[:16], embedder)
            #  add original index to metadata
            for i, doc in enumerate(docs):
                doc.metadata["html_index"] = html_indexes[i]
                doc.metadata["chunk_index"] = i
            if embedder is None:
                retriever=BM25Retriever.from_documents(docs)
                retriever.k = len(docs)
            else:
                batch_size = 256
                # print(f"indexing {len(docs)} chunks")
                try:
                    future = executor.submit(FAISS.from_documents, docs[:batch_size], embedder)
                    db = future.result()
                    for doc_batch_idx in range(batch_size, len(docs), batch_size):
                        # db.add_documents(docs[doc_batch_idx:doc_batch_idx + 16])
                        future = executor.submit(db.add_documents, docs[doc_batch_idx:doc_batch_idx + batch_size])
                        future.result()
                except Exception as e:
                    print(f"batch size {batch_size} failed, try batch size 1")
                    future = executor.submit(FAISS.from_documents, docs[:1], embedder)
                    db = future.result()
                    for doc_batch_idx in range(1, len(docs)):
                        # db.add_documents(docs[doc_batch_idx:doc_batch_idx + 4])
                        future = executor.submit(db.add_documents, docs[doc_batch_idx:doc_batch_idx + 1])
                        future.result()

                retriever = db.as_retriever(search_kwargs={"k": len(docs)})
                # print(f"indexed {len(docs)} chunks")
                # rerank page contents

            # ranked_docs = retriever.invoke(question)
            future = executor.submit(retriever.invoke, question)
            ranked_docs = future.result()
            data_line["page_contents"] = [doc.page_content for doc in ranked_docs]
            data_line["metadatas"] = [doc.metadata for doc in ranked_docs]


        def remove_adjacent_duplicates(input_list):
            if not input_list:  # If the list is empty, return it
                return []
            result = [re.sub(r' +', ' ', re.sub(r'\n+', ' ', input_list[0]))]
            cur_res = input_list[0]
            # Start with the first element
            for item in input_list[1:]:
                if item != cur_res:  # Only add if it's not the same as the last added
                    result.append(re.sub(r' +', ' ', re.sub(r'\n+', ' ', item)))
                    cur_res = item
            return result
    output_dir = f"../../output/{dataset}/{model_name}/{_llm_name}"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/{model_name}-{dataset}.jsonl"
    data_dict = {}
    for l in data_lines:
        data_dict[l["id"]] = {"question": l["question"],
                              "query_time": l["query_time"],
                              "page_contents": remove_adjacent_duplicates(l["page_contents"]),
                              }
    with open(output_file, 'w') as json_file:
        json.dump(data_dict, json_file, indent=4)

    # chat_tokenizer_path = "../../pretrained_model/meta-llama/Llama-3.2-3B-Instruct"
    # chat_tokenizer = AutoTokenizer.from_pretrained(chat_tokenizer_path, trust_remote_code=True)
    # context_windows = ["192k", "128k", "64k", "32k", "16k", "8k", "4k", "2k"]
    # context_windows = ["32k", "16k", "8k", "4k", "2k"]

    # def fill_chunk(context_window, dataset, data_lines):
    #     output_dir = f"../../output/{dataset}/{model_name}/{_llm_name}/"
    #     output_file = f"{output_dir}/{rerank_model}-{dataset}-{context_window}.jsonl"
    #     if not os.path.exists(output_dir):
    #         os.makedirs(output_dir, exist_ok=True)
    #
    #     for idx in tqdm(range(len(data_lines)), desc=f"{dataset} trimming htmls with context window {context_window}"):
    #         #  fill in chunks until the length of the input exceeds the max length
    #         chunks = data_lines[idx]['page_contents']
    #         max_context_window = re.match(r"(\d+)k", context_window).group(1)
    #         max_context_window = int(max_context_window) * 1000
    #
    #         ref_chunks = chunks[:1]
    #         ref_token_length = len(chat_tokenizer.encode(" ".join(ref_chunks), add_special_tokens=False))
    #         for i in range(1, len(chunks)):
    #             if ref_token_length > max_context_window:
    #                 break
    #             ref_chunks.append(chunks[i])
    #             ref_token_length += len(chat_tokenizer.encode(chunks[i], add_special_tokens=False))
    #         if len(ref_chunks) == 1:
    #             ref_chunks[0] = truncate_input(ref_chunks[0], chat_tokenizer, max_context_window)
    #         else:
    #             while True:
    #                 ref_chunks = ref_chunks[:-1]
    #                 total_token_length = len(chat_tokenizer.encode(" ".join(ref_chunks), add_special_tokens=False))
    #                 if total_token_length <= max_context_window:
    #                     break
    #         total_token_length = len(chat_tokenizer.encode(" ".join(ref_chunks), add_special_tokens=False))
    #         assert total_token_length <= max_context_window, f"total token length {total_token_length} exceeds {max_context_window}"
    #
    #         data_lines[idx]['html_trim'] = ref_chunks
    #
    #
    #     def remove_adjacent_duplicates(input_list):
    #         if not input_list:  # If the list is empty, return it
    #             return []
    #         result = [input_list[0]]  # Start with the first element
    #         for item in input_list[1:]:
    #             if item != result[-1]:  # Only add if it's not the same as the last added
    #                 result.append(item)
    #         return result
    #     data_dict = {}
    #     for l in data_lines:
    #         data_dict[l["id"]] = {"html_trim": remove_adjacent_duplicates(l["html_trim"])}
    #     with open(output_file, 'w') as json_file:
    #         json.dump(data_dict, json_file, indent=4)
    #     loguru.logger.info(
    #         f"html trimmed with context window {context_window}")
    #     loguru.logger.info(f"saved to {output_file}")
    #
    # # max_processes = 8
    # # process_pool=[]
    # # for context_window in context_windows:
    # #     for dataset in datasets:
    # #         p = multiprocessing.Process(target=fill_chunk, args=(context_window, dataset, data_lines))
    # #         p.start()
    # #         process_pool.append(p)
    # #
    # #     if len(process_pool) >= max_processes:
    # #         for process in process_pool:
    # #             process.join()
    # #         process_pool = []
    # #
    # # if len(process_pool) > 0:
    # #     for process in process_pool:
    # #         process.join()
    # fill_chunk(context_window, dataset, data_lines)
    loguru.logger.info("All processes finished")

