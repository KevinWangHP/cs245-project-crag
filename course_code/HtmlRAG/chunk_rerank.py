import argparse
import concurrent.futures
import json
import os
import pathlib
import re

from langchain_community.embeddings import BaichuanTextEmbeddings, HuggingFaceHubEmbeddings, HuggingFaceBgeEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from tqdm import tqdm
from typing import List
import sys

from transformers import AutoTokenizer

sys.path.append("./")
from html4rag.html_utils import HTMLSplitter, headers_to_split_on, clean_xml

html_splitter = HTMLSplitter(headers_to_split_on=headers_to_split_on)


class TEIEmbeddings(HuggingFaceHubEmbeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # replace newlines, which can negatively affect performance.
        texts = [text.replace("\n", " ") for text in texts]
        #  truncate to 1024 tokens approximately
        for i in range(len(texts)):
            text = texts[i]
            words = text.split(" ")
            if len(words) > 1024 or len(text) > 1024:
                tokens = tokenizer.encode(text, add_special_tokens=False)
                tokens = tokens[:4096]
                texts[i] = tokenizer.decode(tokens)

        _model_kwargs = self.model_kwargs or {}
        try:
            #  api doc: https://huggingface.github.io/text-embeddings-inference/#/Text%20Embeddings%20Inference/embed
            responses = self.client.post(
                json={"inputs": texts, **_model_kwargs}, task=self.task
            )
        except Exception as e:
            print(f"error: {e}")
            json.dump(texts, open("error_texts.json", "w"))
        return json.loads(responses.decode())


class VLLMEmbeddings(HuggingFaceHubEmbeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # replace newlines, which can negatively affect performance.
        texts = [text.replace("\n", " ") for text in texts]
        #  truncate to 1024 tokens approximately
        for i in range(len(texts)):
            text = texts[i]
            words = text.split(" ")
            if len(words) > 1024:
                text = " ".join(words[:1024])
                texts[i] = text

        _model_kwargs = self.model_kwargs or {}
        try:
            #  api doc: https://huggingface.github.io/text-embeddings-inference/#/Text%20Embeddings%20Inference/embed
            responses = self.client.post(
                json={"inputs": texts, **_model_kwargs}, task=self.task
            )
        except Exception as e:
            print(f"error: {e}")
            json.dump(texts, open("error_texts.json", "w"))
        return json.loads(responses.decode())


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dataset", type=str, default="kdd_crag")
    argparser.add_argument("--split", type=str, default="test")
    argparser.add_argument("--search_engine", type=str, default="bing")
    argparser.add_argument("--mini_dataset", action="store_true")
    argparser.add_argument("--rerank_model", type=str, default="bgelargeen")
    argparser.add_argument("--url", type=str, default="../../pretrained_model/BAAI/bge-large-en")
    argparser.add_argument("--rewrite_method", type=str, default="slimplmqr")
    argparser.add_argument("--device", type=str, default="cuda:2")
    args = argparser.parse_args()

    dataset = args.dataset
    split = args.split
    search_engine = args.search_engine
    mini_dataset = True
    rerank_model = args.rerank_model
    rewrite_method = args.rewrite_method
    url = args.url


    # if rerank_model == "bc":
    #     embedder = BaichuanTextEmbeddings(baichuan_api_key="sk-1d50105f21bd6263265fcaedfdedd1d4")
    # elif rerank_model == "bgelargeen":
    #     embedder = TEIEmbeddings(
    #         model=url,
    #         huggingfacehub_api_token="a-default-token",
    #         model_kwargs={"truncate": True})
    # elif rerank_model == "e5-mistral":
    #     embedder = TEIEmbeddings(
    #         model=url,
    #         huggingfacehub_api_token="a-default-token",
    #         model_kwargs={"truncate": True})
    # elif rerank_model == "bm25":
    #     embedder = None
    # else:
    #     raise NotImplementedError(f"rerank model {rerank_model} not implemented")

    embedder = HuggingFaceBgeEmbeddings(
        model_name = url,
        model_kwargs={'device': args.device},
        encode_kwargs={'normalize_embeddings': True},
    )

    tokenizer_path = "../../pretrained_model/BAAI/bge-large-en"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    input_file = f"./html_data/{dataset}/{dataset}.jsonl"
    output_file = f"./html_data/{dataset}/chunk_rerank/{rerank_model}-{dataset}-{split}.jsonl"
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))


    if rerank_model == "bgelargeen":
        query_instruction_for_retrieval = "Represent this sentence for searching relevant passages: "
    elif rerank_model == "e5-mistral":
        query_instruction_for_retrieval = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
    elif rerank_model == "bm25":
        query_instruction_for_retrieval = ""
    else:
        raise NotImplementedError(f"rerank model {rerank_model} not implemented")

    print(f"input_file: {input_file}")
    with open(input_file, "r") as file:
        data_lines = json.load(file)

    if args.mini_dataset:
        data_lines = data_lines[:10]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for idx, data_line in tqdm(enumerate(data_lines), total=len(data_lines)):
            question = query_instruction_for_retrieval + data_line["question"]
            docs = []
            html = [clean_xml(d) for d in data_lines[idx][f'search_results']]
            html = [h for h in html if h.strip()]
            html_indexes = []
            for idh, h in enumerate(html):
                splits = html_splitter.split_text(h)
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

    # save to file
    with open(output_file, "w") as f:
        for data_line in data_lines:
            f.write(json.dumps(data_line, ensure_ascii=False) + "\n")
