import argparse
import concurrent.futures
import json
import os
import threading

import bs4
import loguru
from langchain_community.embeddings import HuggingFaceHubEmbeddings
from langchain_community.vectorstores import FAISS
from tqdm import tqdm
from typing import List
from langchain_core.documents import Document
import sys

from transformers import AutoTokenizer

sys.path.append("./")
from html4rag.html_utils import split_tree, clean_xml


class TEIEmbeddings(HuggingFaceHubEmbeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # replace newlines, which can negatively affect performance.
        texts = [text.replace("\n", " ") for text in texts]
        #  truncate to 1024 tokens approximately
        for i in range(len(texts)):
            text = texts[i]
            words = text.split(" ")
            if len(words) > 1024 or len(text) > 1024:
                tokens=tokenizer.encode(text, add_special_tokens=False)
                tokens=tokens[:4096]
                texts[i]=tokenizer.decode(tokens)
            texts[i] = texts[i].strip()
            if not texts[i]:
                texts[i] = "Some padding text"

        _model_kwargs = self.model_kwargs or {}
        try:
            #  api doc: https://huggingface.github.io/text-embeddings-inference/#/Text%20Embeddings%20Inference/embed
            responses = self.client.post(
                json={"inputs": texts, **_model_kwargs}, task=self.task
            )
        except Exception as e:
            print(f"error: {e}, texts: {texts}")
        return json.loads(responses.decode())


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--search_engine", type=str, default="bing")
    argparser.add_argument("--dataset", type=str, default="asqa")
    argparser.add_argument("--split", type=str, default="test")
    argparser.add_argument("--rewrite_method", type=str, default="slimplmqr")
    argparser.add_argument("--rerank_model", type=str, default="bgelargeen")
    argparser.add_argument("--url", type=str, default="http://bge-large-en.search.cls-3nbemh6i.ml.baichuan-inc.com")
    argparser.add_argument("--mini_dataset", action="store_true")
    argparser.add_argument("--max_node_words", type=int, default=128)
    args = argparser.parse_args()
    split = args.split
    search_engine = args.search_engine
    rewrite_method = args.rewrite_method
    rerank_model = args.rerank_model
    dataset = args.dataset
    max_node_words = args.max_node_words

    thread_pool = []
    if rerank_model == "bgelargeen":
        query_instruction_for_retrieval = "Represent this sentence for searching relevant passages: "
    elif rerank_model == "e5-mistral":
        query_instruction_for_retrieval = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
    else:
        raise NotImplementedError(f"rerank model {rerank_model} not implemented")
    tokenizer_path = "../../huggingface/e5-mistral-7b-instruct"
    tokenizer=AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    embedder = TEIEmbeddings(
        model=args.url,
        huggingfacehub_api_token="a-default-token",
        model_kwargs={"truncate": True})

    data_file = f"./html_data/{dataset}/{search_engine}/{search_engine}html-{rewrite_method}-{dataset}-simple-{split}.jsonl"
    data_lines = [json.loads(line) for line in open(data_file)]
    loguru.logger.info(f"Reading data from {data_file}")
    loguru.logger.info(f"max_node_words: {max_node_words}")
    if args.mini_dataset:
        data_lines = data_lines[:10]

    output_file = f"./html_data/{dataset}/tree-rerank/{search_engine}html-{rewrite_method}-{rerank_model}-{max_node_words}-{dataset}-{split}.jsonl"
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for nidx in tqdm(range(len(data_lines)), total=len(data_lines), desc=f"{dataset}:"):
            htmls = [clean_xml(d['html']) for d in data_lines[nidx][f'{rewrite_method}_results']]
            htmls = [h for h in htmls if h.strip()]
            soup = bs4.BeautifulSoup("", 'html.parser')
            for html in htmls:
                soup.append(bs4.BeautifulSoup(html, 'html.parser'))
            split_res = split_tree(soup, max_node_words=max_node_words)
            path_tags = [res[0] for res in split_res]
            paths = [res[1] for res in split_res]
            is_leaf = [res[2] for res in split_res]
            question = query_instruction_for_retrieval + data_lines[nidx]['question']

            node_docs = []
            for pidx in range(len(paths)):
                node_docs.append(Document(page_content=path_tags[pidx].get_text(), metadata={"path_idx": pidx}))
            batch_size = 256
            # db = FAISS.from_documents(node_docs[:batch_size], embedder)
            future = executor.submit(FAISS.from_documents, node_docs[:batch_size], embedder)
            db = future.result()
            if len(node_docs) > batch_size:
                for doc_batch_idx in range(batch_size, len(node_docs), batch_size):
                    # db.add_documents(node_docs[doc_batch_idx:doc_batch_idx + batch_size])
                    future = executor.submit(db.add_documents, node_docs[doc_batch_idx:doc_batch_idx + batch_size])
                    future.result()
            retriever = db.as_retriever(search_kwargs={"k": len(node_docs)})
            # ranked_docs = retriever.invoke(question)
            future = executor.submit(retriever.invoke, question)
            ranked_docs = future.result()
            path_rankings = [doc.metadata["path_idx"] for doc in ranked_docs]

            data_lines[nidx]["path_rankings"] = path_rankings
            data_lines[nidx]["html"] = str(soup)
            data_lines[nidx]["paths"] = paths
            data_lines[nidx]["is_leaf"] = is_leaf

    with open(output_file, "w") as f:
        for idx in range(len(data_lines)):
            f.write(json.dumps(data_lines[idx], ensure_ascii=False) + "\n")
    loguru.logger.info(f"Saved parsed html to {output_file}")

