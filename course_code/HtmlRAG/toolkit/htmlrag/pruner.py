
from langchain_core.documents import Document
import bs4
from .html_utils import trim_path, simplify_html, truncate_input, TokenIdNode

import json
from typing import List, Tuple

import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch


class Pruner:

    def calculate_block_rankings(self, question: str, html: str, block_tree: List[Tuple]):
        raise NotImplementedError("calculate_block_rankings method is not implemented")

    def prune_HTML(self, html, block_tree: List[Tuple], block_rankings: List[int], chat_tokenizer, max_context_window: int):
        paths = [b[1] for b in block_tree]
        is_leaf = [b[2] for b in block_tree]

        paths = [{"path": paths[i], "is_leaf": is_leaf[i]} for i in range(len(paths))]
        for idj in range(len(paths)):
            path_idx = int(block_rankings[idj])
            paths[path_idx]["ranking"] = idj

        soup = bs4.BeautifulSoup(html, 'html.parser')
        #  sort paths by ranking
        paths = sorted(paths, key=lambda x: x["ranking"])
        total_token_length = 0
        for idj in range(len(paths)):
            path = paths[idj]["path"]
            tag = soup
            for p in path:
                for child in tag.contents:
                    if isinstance(child, bs4.element.Tag):
                        if child.name == p:
                            tag = child
                            break
            assert tag.name == path[-1], f"tag name: {tag.name}, path[-1]: {path[-1]}"
            paths[idj]["tag"] = tag
            paths[idj]["token_length"] = len(chat_tokenizer.encode(str(tag), add_special_tokens=False))
            total_token_length += paths[idj]["token_length"]
            if total_token_length > max_context_window:
                paths = paths[:idj+1]
                break

        #  remove low ranking paths
        while total_token_length > max_context_window:
            if len(paths) == 1:
                break
            discarded_path = paths.pop()
            total_token_length -= discarded_path["token_length"]
            trim_path(discarded_path)

        total_token_length = len(chat_tokenizer.encode(simplify_html(soup), add_special_tokens=False))
        while total_token_length > max_context_window:
            if len(paths) == 1:
                break
            discarded_path = paths.pop()
            trim_path(discarded_path)
            total_token_length = len(chat_tokenizer.encode(simplify_html(soup), add_special_tokens=False))

        if total_token_length > max_context_window:
            html_trim = truncate_input(simplify_html(soup), chat_tokenizer, max_context_window)
        else:
            html_trim = simplify_html(soup)

        assert len(chat_tokenizer.encode(
            html_trim,
            add_special_tokens=False)) <= max_context_window, f"html length: {len(chat_tokenizer.encode(html_trim, add_special_tokens=False))}, max_context_window: {max_context_window}"

        return html_trim


class EmbedHTMLPruner(Pruner):

    def __init__(self, embed_model="BAAI/bge-large-en", local_inference=True, query_instruction_for_retrieval="", endpoint=""):
        self.query_instruction_for_retrieval = ""
        if embed_model == "BAAI/bge-large-en":
            self.query_instruction_for_retrieval = "Represent this sentence for searching relevant passages: "
        elif embed_model == "intfloat/e5-mistral-7b-instruct":
            self.query_instruction_for_retrieval = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
        if query_instruction_for_retrieval:
            self.query_instruction_for_retrieval = query_instruction_for_retrieval

        if local_inference:
            from langchain_huggingface import HuggingFaceEmbeddings
            self.embedder = HuggingFaceEmbeddings(model_name=embed_model)
        else:
            from langchain_huggingface import HuggingFaceEndpointEmbeddings
            embedder = HuggingFaceEndpointEmbeddings(
                model=endpoint,
                huggingfacehub_api_token="a-default-token",
                model_kwargs={"truncate": True})
            self.embedder = embedder


    def calculate_block_rankings(self, question: str, html: str, block_tree: List[Tuple]):
        question = self.query_instruction_for_retrieval + question
        path_tags = [b[0] for b in block_tree]
        paths = [b[1] for b in block_tree]

        node_docs = []
        for pidx in range(len(paths)):
            node_docs.append(Document(page_content=path_tags[pidx].get_text(), metadata={"path_idx": pidx}))
        batch_size = 256

        from langchain_community.vectorstores import FAISS
        db = FAISS.from_documents(node_docs[:batch_size], self.embedder)
        if len(node_docs) > batch_size:
            for doc_batch_idx in range(batch_size, len(node_docs), batch_size):
                db.add_documents(node_docs[doc_batch_idx:doc_batch_idx + batch_size])
        retriever = db.as_retriever(search_kwargs={"k": len(node_docs)})
        ranked_docs = retriever.invoke(question)
        block_rankings = [doc.metadata["path_idx"] for doc in ranked_docs]

        return block_rankings


class BM25HTMLPruner(Pruner):
    def calculate_block_rankings(self, question: str, html: str, block_tree: List[Tuple]):
        path_tags = [b[0] for b in block_tree]
        paths = [b[1] for b in block_tree]

        node_docs = []
        for pidx in range(len(paths)):
            node_docs.append(Document(page_content=path_tags[pidx].get_text(), metadata={"path_idx": pidx}))
        from langchain_community.retrievers import BM25Retriever
        retriever = BM25Retriever.from_documents(documents=node_docs)
        retriever.k = len(node_docs)
        ranked_docs = retriever.invoke(question)
        block_rankings = [doc.metadata["path_idx"] for doc in ranked_docs]

        return block_rankings


class GenHTMLPruner(Pruner):

    def __init__(self,  gen_model, max_node_words, device="cpu"):
        self.gen_model = AutoModelForSeq2SeqLM.from_pretrained(gen_model, trust_remote_code=True, torch_dtype=torch.bfloat16).eval()
        self.gen_model.to(device)
        self.gen_model.max_node_words = max_node_words
        self.node_tokenizer = AutoTokenizer.from_pretrained(gen_model, trust_remote_code=True)
        self.log_threshold = 1e-9

    def calculate_block_rankings(self, question: str, html: str, block_tree: List[Tuple]=None):
        log_threshold = self.log_threshold
        html_res = self.gen_model.generate_html_tree(self.node_tokenizer, [question], [html])
        node_tree=html_res[0]["node_tree"]
        paths=html_res[0]["paths"]

        root = TokenIdNode(-1, prob=1.0, input_ids=[])
        for nidx, line in enumerate(node_tree):
            node, prob, input_ids = line.split("|")
            node = int(node)
            prob = float(prob)
            input_ids = json.loads(input_ids)
            if node == -1:
                # root node
                pass
            else:
                # token = node_tokenizer.decode([node])
                parent = root
                for tid in input_ids:
                    find_child = False
                    for c in parent.children:
                        if c.input_ids[-1] == tid:
                            parent = c
                            find_child = True
                            break
                    if not find_child:
                        break
                node = TokenIdNode(node, parent=parent, prob=prob, input_ids=input_ids)

        #  sort paths by prob
        path_probs = []
        for pidx in range(len(paths)):
            path = paths[pidx]
            str_path = "<" + "><".join(path) + ">"
            path_tokens = self.node_tokenizer.encode(str_path, add_special_tokens=False)
            path_token_prob = []
            node = root
            for tid in path_tokens:
                for c in node.children:
                    if c.name == tid:
                        node = c
                        # print(f"token id: {tid}, token: {node_tokenizer.decode([tid])}, prob: {node.prob}")
                        path_token_prob.append(node.prob)
                        break
            path_probs.append(path_token_prob)

        average_depth = np.mean([len(p) for p in paths])
        path_log_probs = [
            np.sum([np.log(p) if p > log_threshold else np.log(log_threshold) for p in path]) / average_depth
            for path in path_probs]

        path_rankings = np.argsort(path_log_probs)[::-1]
        return list(path_rankings)
