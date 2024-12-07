import argparse
import concurrent.futures
import json
import os
import threading
import torch
import bs4
import loguru
from langchain_community.embeddings import HuggingFaceHubEmbeddings, HuggingFaceEmbeddings, HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from tqdm import tqdm
from typing import List
from langchain_core.documents import Document
import sys
from transformers import AutoTokenizer, AutoModel
from langchain.embeddings.base import Embeddings
sys.path.append("../")
from html4rag.html_utils import split_tree, clean_xml

# def format_prompts(query):
#     system_prompt = "You are provided with a question and various references. Your task is to answer the question succinctly, using the fewest words possible. If the references do not contain the necessary information to answer the question, respond with 'I don't know'. There is no need to explain the reasoning behind your answers."
#     formatted_prompts = []
#
#     for _idx, query in enumerate(queries):
#         query_time = query_times[_idx]
#         retrieval_results = batch_retrieval_results[_idx]
#
#         user_message = ""
#         references = ""
#
#         if len(retrieval_results) > 0:
#             references += "# References \n"
#             # Format the top sentences as references in the model's prompt template.
#             for _snippet_idx, snippet in enumerate(retrieval_results):
#                 references += f"- {snippet.strip()}\n"
#
#         references = references[:MAX_CONTEXT_REFERENCES_LENGTH]
#         # Limit the length of references to fit the model's input size.
#
#         user_message += f"{references}\n------\n\n"
#         user_message += f"Using only the references listed above, answer the following question: \n"
#         user_message += f"Current Time: {query_time}\n"
#         user_message += f"Question: {query}\n"
#
#         if self.is_server:
#             # there is no need to wrap the messages into chat when using the server
#             # because we use the chat API: chat.completions.create
#             formatted_prompts.append(
#                 [
#                     {"role": "system", "content": system_prompt},
#                     {"role": "user", "content": user_message},
#                 ]
#             )
#         else:
#             formatted_prompts.append(
#                 self.tokenizer.apply_chat_template(
#                     [
#                         {"role": "system", "content": system_prompt},
#                         {"role": "user", "content": user_message},
#                     ],
#                     tokenize=False,
#                     add_generation_prompt=True,
#                 )
#             )


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
    argparser.add_argument("--dataset", type=str, default="kdd_crag")
    argparser.add_argument("--split", type=str, default="test")
    argparser.add_argument("--rerank_model", type=str, default="bgelargeen")
    argparser.add_argument("--url", type=str, default="https://api-inference.huggingface.co/models/BAAI/bge-large-en")
    argparser.add_argument("--max_node_words", type=int, default=128)
    argparser.add_argument("--device", type=str, default="cuda:1")
    args = argparser.parse_args()
    split = args.split
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
    model_name = "../../pretrained_model/BAAI/bge-large-en"
    tokenizer=AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    embedder = HuggingFaceBgeEmbeddings(
        model_name = "sentence-transformers/all-MiniLM-l6-v2",
        model_kwargs={'device': args.device},
        encode_kwargs={'normalize_embeddings': True},
    )

    data_file = f"./html_data/{dataset}/{dataset}.jsonl"
    with open(data_file, "r", encoding="utf-8") as json_file:
        data_lines = json.load(json_file)
    loguru.logger.info(f"Reading data from {data_file}")
    loguru.logger.info(f"max_node_words: {max_node_words}")

    output_file = f"./html_data/{dataset}/tree-rerank/{max_node_words}-{dataset}-{split}.jsonl"
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    for nidx in tqdm(range(len(data_lines)), total=len(data_lines), desc=f"{dataset}:"):
        htmls = [clean_xml(d) for d in data_lines[nidx][f'search_results']]
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

        db = FAISS.from_documents(node_docs[:batch_size], embedder)
        if len(node_docs) > batch_size:
            for doc_batch_idx in range(batch_size, len(node_docs), batch_size):
                db.add_documents(node_docs[doc_batch_idx:doc_batch_idx + batch_size])

        retriever = db.as_retriever(search_kwargs={"k": len(node_docs)})

        ranked_docs = retriever.invoke(question)
        path_rankings = [doc.metadata["path_idx"] for doc in ranked_docs]

        data_lines[nidx]["path_rankings"] = path_rankings
        data_lines[nidx]["html"] = str(soup)
        data_lines[nidx]["paths"] = paths
        data_lines[nidx]["is_leaf"] = is_leaf

    with open(output_file, "w") as f:
        for idx in range(len(data_lines)):
            f.write(json.dumps(data_lines[idx], ensure_ascii=False) + "\n")
    loguru.logger.info(f"Saved parsed html to {output_file}")

