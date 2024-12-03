import os
import re
from abc import ABC
from collections import defaultdict
from typing import Any, Dict, List
from openai import OpenAI
import numpy as np
import ray
import torch
import vllm
from blingfire import text_to_sentences_and_offsets
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from multifeature.main_content_extractor import MainContentExtractor
from multifeature.bm25_retriever import Bm25Retriever
from multifeature.faiss_retriever import FaissRetriever
from multifeature.rerank_model import reRankLLM
from multifeature.utils import trim_predictions_to_max_token_length
from blingfire import text_to_sentences_and_offsets
import json

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoModel,
    pipeline,
    AutoModelForSequenceClassification
)
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.schema.embeddings import Embeddings

RERANKER_MODEL_PATH = "../pretrained_model/BAAI/bge-reranker-v2-gemma"
EMBEDDING_MODEL_PATH = "../pretrained_model/BAAI/bge-large-en"
LLAMA3_MODEL_PATH = "../pretrained_model/meta-llama/Llama-3.2-3B-Instruct"
#### CONFIG PARAMETERS ---

# Define the number of context sentences to consider for generating an answer.
NUM_CONTEXT_SENTENCES = 20
# Set the maximum length for each context sentence (in characters).
MAX_CONTEXT_SENTENCE_LENGTH = 1000
# Set the maximum context references length (in characters).
MAX_CONTEXT_REFERENCES_LENGTH = 4000

# Batch size you wish the evaluators will use to call the `batch_generate_answer` function
AICROWD_SUBMISSION_BATCH_SIZE = 1  # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.

# VLLM Parameters
VLLM_TENSOR_PARALLEL_SIZE = 2  # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.
VLLM_GPU_MEMORY_UTILIZATION = 0.79  # 0.85  # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.

# Sentence Transformer Parameters
SENTENTENCE_TRANSFORMER_BATCH_SIZE = 128  # TUNE THIS VARIABLE depending on the size of your embedding model and GPU mem available

class RAGModel:
    """
    An example RAGModel for the KDDCup 2024 Meta CRAG Challenge
    which includes all the key components of a RAG lifecycle.
    """

    def __init__(self, llm_name="meta-llama/Llama-3.2-3B-Instruct", is_server=False, vllm_server=None, device="cuda"):
        self.device = device
        self.initialize_models(llm_name, is_server, vllm_server)
        self.dataset = "example_data"
        self.path = f'../output/{self.dataset}/multifeature/Llama-3.2-3B-Instruct/'
        # self.reranker = FlagLLMReranker(RERANKER_MODEL_PATH, use_fp16=True, device=["cuda:1"])  # Setting use_fp16 to True speeds up computation with a slight performance degradation
        self.model_tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_PATH)
        self.model = AutoModelForCausalLM.from_pretrained(RERANKER_MODEL_PATH)
        self.retrieval_dict = []
        with open(self.path + f"bgelargeen-{self.dataset}.jsonl", 'r') as file:
            self.retrieval_dict.append(json.load(file))
        with open(self.path + f"bm25-{self.dataset}.jsonl", 'r') as file:
            self.retrieval_dict.append(json.load(file))

        self.top_k = [5, 5]
        self.rerank_top_k = 5
        self.max_ctx_sentence_length = 200

        self.overlap_length = 200
        self.window_size = 500

        self.sim_threshold = 0.75

    def initialize_models(self, llm_name, is_server, vllm_server):
        self.llm_name = llm_name
        self.is_server = is_server
        self.vllm_server = vllm_server

        if self.is_server:
            # initialize the model with vllm server
            openai_api_key = "EMPTY"
            openai_api_base = self.vllm_server
            self.llm_client = OpenAI(
                api_key=openai_api_key,
                base_url=openai_api_base,
            )
        else:
            if not os.path.exists(self.llm_name):
                raise Exception(
                    f"""
                The evaluators expect the model weights to be checked into the repository,
                but we could not find the model weights at {self.llm_name}

                Please follow the instructions in the docs below to download and check in the model weights.

                https://gitlab.aicrowd.com/aicrowd/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024/meta-comphrehensive-rag-benchmark-starter-kit/-/blob/master/docs/dataset.md
                """
                )
            # initialize the model with vllm offline inference
            self.llm = vllm.LLM(
                model=self.llm_name,
                worker_use_ray=True,
                tensor_parallel_size=VLLM_TENSOR_PARALLEL_SIZE,
                gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
                trust_remote_code=True,
                dtype="half",  # note: bfloat16 is not supported on nvidia-T4 GPUs
                enforce_eager=True
            )
            self.tokenizer = self.llm.get_tokenizer()

        # Load a sentence transformer model optimized for sentence embeddings, using CUDA if available.
        # self.sentence_model = SentenceTransformer(
        #     "all-MiniLM-L6-v2",
        #     device=torch.device(
        #         self.device
        #     )
        # )

    # def calculate_embeddings(self, sentences):
    #     """
    #     Compute normalized embeddings for a list of sentences using a sentence encoding model.
    #
    #     This function leverages multiprocessing to encode the sentences, which can enhance the
    #     processing speed on multi-core machines.
    #
    #     Args:
    #         sentences (List[str]): A list of sentences for which embeddings are to be computed.
    #
    #     Returns:
    #         np.ndarray: An array of normalized embeddings for the given sentences.
    #
    #     """
    #     embeddings = self.sentence_model.encode(
    #         sentences=sentences,
    #         normalize_embeddings=True,
    #         batch_size=SENTENTENCE_TRANSFORMER_BATCH_SIZE,
    #     )
    #     # Note: There is an opportunity to parallelize the embedding generation across 4 GPUs
    #     #       but sentence_model.encode_multi_process seems to interefere with Ray
    #     #       on the evaluation servers.
    #     #       todo: this can also be done in a Ray native approach.
    #     #
    #     return embeddings

    def post_process(self, answer):
        if "i don't know" in answer.lower():
            return "i don't know"
        else:
            return answer

    def get_batch_size(self) -> int:
        """
        Determines the batch size that is used by the evaluator when calling the `batch_generate_answer` function.

        The evaluation timeouts linearly scale with the batch size.
            i.e.: time out for the `batch_generate_answer` call = batch_size * per_sample_timeout


        Returns:
            int: The batch size, an integer between 1 and 16. It can be dynamic
                 across different batch_generate_answer calls, or stay a static value.
        """
        self.batch_size = AICROWD_SUBMISSION_BATCH_SIZE
        return self.batch_size

    def reranking(self, input_pair) -> List[float]:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        def get_inputs(pairs, tokenizer, prompt=None, max_length=1024):
            if prompt is None:
                prompt = "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'."
            sep = "\n"
            prompt_inputs = tokenizer(prompt,
                                      return_tensors=None,
                                      add_special_tokens=False)['input_ids']
            sep_inputs = tokenizer(sep,
                                   return_tensors=None,
                                   add_special_tokens=False)['input_ids']
            inputs = []
            for query, passage in pairs:
                query_inputs = tokenizer(f'A: {query}',
                                         return_tensors=None,
                                         add_special_tokens=False,
                                         max_length=max_length * 3 // 4,
                                         truncation=True)
                passage_inputs = tokenizer(f'B: {passage}',
                                           return_tensors=None,
                                           add_special_tokens=False,
                                           max_length=max_length,
                                           truncation=True)
                item = tokenizer.prepare_for_model(
                    [tokenizer.bos_token_id] + query_inputs['input_ids'],
                    sep_inputs + passage_inputs['input_ids'],
                    truncation='only_second',
                    max_length=max_length,
                    padding=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                    add_special_tokens=False
                )
                item['input_ids'] = item['input_ids'] + sep_inputs + prompt_inputs
                item['attention_mask'] = [1] * len(item['input_ids'])
                inputs.append(item)
            return tokenizer.pad(
                inputs,
                padding=True,
                max_length=max_length + len(sep_inputs) + len(prompt_inputs),
                pad_to_multiple_of=8,
                return_tensors='pt',
            )


        yes_loc = self.model_tokenizer('Yes', add_special_tokens=False)['input_ids'][0]
        self.model.eval()

        with torch.no_grad():
            inputs = get_inputs(input_pair, self.model_tokenizer)
            scores = self.model(**inputs, return_dict=True).logits[:, -1, yes_loc].view(-1, ).float()
        return scores


    def batch_generate_answer(self, batch: Dict[str, Any]) -> List[str]:
        """
        Generates answers for a batch of queries using associated (pre-cached) search results and query times.

        Parameters:
            batch (Dict[str, Any]): A dictionary containing a batch of input queries with the following keys:
                - 'interaction_id;  (List[str]): List of interaction_ids for the associated queries
                - 'query' (List[str]): List of user queries.
                - 'search_results' (List[List[Dict]]): List of search result lists, each corresponding
                                                      to a query. Please refer to the following link for
                                                      more details about the individual search objects:
                                                      https://gitlab.aicrowd.com/aicrowd/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024/meta-comphrehensive-rag-benchmark-starter-kit/-/blob/master/docs/dataset.md#search-results-detail
                - 'query_time' (List[str]): List of timestamps (represented as a string), each corresponding to when a query was made.

        Returns:
            List[str]: A list of plain text responses for each query in the batch. Each response is limited to 75 tokens.
            If the generated response exceeds 75 tokens, it will be truncated to fit within this limit.

        Notes:
        - If the correct answer is uncertain, it's preferable to respond with "I don't know" to avoid
          the penalty for hallucination.
        - Response Time: Ensure that your model processes and responds to each query within 30 seconds.
          Failing to adhere to this time constraint **will** result in a timeout during evaluation.
        """
        batch_interaction_ids = batch["interaction_id"]
        queries = batch["query"]
        batch_search_results = batch["search_results"]
        query_times = batch["query_time"]


        batch_retrieval_results = []
        for _idx, interaction_id in enumerate(batch_interaction_ids):
            query = queries[_idx]
            query_time = query_times[_idx]
            retrieval_results = []
            for i in range(len(self.retrieval_dict)):
                retrieval_results.extend(self.retrieval_dict[i][interaction_id]["page_contents"][:self.top_k[i]])

            retrieval_results = list(set(retrieval_results))
            retrieval_reranking = [[query, retrieval_results[i]] for i in range(len(retrieval_results))]
            scores = self.reranking(retrieval_reranking)

            combined = list(zip(scores, retrieval_results))
            # Sort by scores in descending order
            sorted_combined = sorted(combined, key=lambda x: x[0], reverse=True)
            # Get the top k results
            top_k = sorted_combined[:self.rerank_top_k]
            # Separate the top_k into scores and results if needed
            rerank_res = [item[1] for item in top_k]
            batch_retrieval_results.append(rerank_res)
        # Prepare formatted prompts from the LLM
        # formatted_prompts = self.merge_format_prompts(queries, query_times, batch_retrieval_results)
        # formatted_prompts = self.format_prompts(queries, query_times, batch_retrieval_results)
        formatted_prompts = self.five_shot_template(queries, query_times, batch_retrieval_results)
        # Generate responses via vllm
        if self.is_server:
            response = self.llm_client.chat.completions.create(
                model=self.llm_name,
                messages=formatted_prompts[0],
                n=2,  # Number of output sequences to return for each prompt.
                top_p=0.6,  # Float that controls the cumulative probability of the top tokens to consider.
                temperature=0.1,  # randomness of the sampling
                # skip_special_tokens=True,  # Whether to skip special tokens in the output.
                max_tokens=300,  # Maximum number of tokens to generate per output sequence.
            )
            answers = [response.choices[0].message.content]
        else:
            responses = self.llm.generate(
                formatted_prompts,
                vllm.SamplingParams(
                    n=1,  # Number of output sequences to return for each prompt.
                    top_p=0.9,  # Float that controls the cumulative probability of the top tokens to consider.
                    temperature=0.1,  # randomness of the sampling
                    skip_special_tokens=True,  # Whether to skip special tokens in the output.
                    max_tokens=50,  # Maximum number of tokens to generate per output sequence.
                ),
                use_tqdm=False
            )

        # Aggregate answers into List[str]
            answers = []
            for response in responses:
                trimmed_answer = self.post_process(trim_predictions_to_max_token_length(response.outputs[0].text))
                print("answer: " + trimmed_answer + "\n")
                answers.append(trimmed_answer)

        return answers


    def five_shot_template(self, queries, query_times, batch_retrieval_results=[]):
        """
        Formats queries, corresponding query_times and retrieval results using the chat_template of the model.

        Parameters:
        - queries (List[str]): A list of queries to be formatted into prompts.
        - query_times (List[str]): A list of query_time strings corresponding to each query.
        - batch_retrieval_results (List[str])
        """
        system_prompt = "You are given a quesition and references which may or may not help answer the question." \
                        "Your goal is to answer the question in as few words as possible and still accurate." \
                        "If the information in the references does not include an answer, you must say 'I don't know'. " \
                        "You must not say any incorrect information. Especially for the time, location, name, and number, you confirm them repeatedly, otherwise you must say 'I don't know'." \
                        "If you cannot be 100% certain that you are right, please answer that I do not know." \
                        "There is no need to explain the reasoning behind your answers.\n"

        formatted_prompts = []

        for _idx, query in enumerate(queries):
            query_time = query_times[_idx]
            retrieval_results = batch_retrieval_results[_idx]

            user_message = ""
            references = ""

            if len(retrieval_results) > 0:
                references += "# References \n"
                # Format the top sentences as references in the model's prompt template.
                for _snippet_idx, snippet in enumerate(retrieval_results):
                    snippet = snippet.replace("\n", " ")
                    references += f"- {snippet.strip()}\n"

            references = references[:MAX_CONTEXT_REFERENCES_LENGTH]
            # Limit the length of references to fit the model's input size.
            user_message += f"{references}\n------\n\n"
            user_message += f"Using only the references listed above, answer the following question: \n"
            user_message += f"Current Time: {query_time}\n"
            user_message += f"Question: {query}\n"
            if self.is_server:
                # there is no need to wrap the messages into chat when using the server
                # because we use the chat API: chat.completions.create
                formatted_prompts.append(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ]
                )
            else:
                formatted_prompts.append(
                    self.tokenizer.apply_chat_template(
                        [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message},
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )

        return formatted_prompts

    def merge_format_prompts(self, queries, query_times, batch_retrieval_results=[]):
        """
        Formats queries, corresponding query_times and retrieval results using the chat_template of the model.

        Parameters:
        - queries (List[str]): A list of queries to be formatted into prompts.
        - query_times (List[str]): A list of query_time strings corresponding to each query.
        - batch_retrieval_results (List[str])
        """
        system_prompt = "You are provided with a question and various references. Your task is to answer the question succinctly, using the fewest words possible. If the references do not contain the necessary information to answer the question, respond with 'I don't know'. You must not say any incorrect information, Especially for the times, location, names, number, and what happened today. If you are not 100% sure please answer 'I don't know'. There is no need to explain the reasoning behind your answers."
        formatted_prompts = []

        for _idx, query in enumerate(queries):
            query_time = query_times[_idx]
            retrieval_results = batch_retrieval_results[_idx]

            user_message = ""
            references = ""

            if len(retrieval_results) > 0:
                references += "# References \n"
                # Format the top sentences as references in the model's prompt template.
                for _snippet_idx, snippet in enumerate(retrieval_results):
                    snippet = snippet.replace("\n", " ")
                    references += f"- {snippet.strip()}\n"

            references = references[:MAX_CONTEXT_REFERENCES_LENGTH]
            # Limit the length of references to fit the model's input size.

            user_message += f"{references}\n------\n\n"
            user_message += f"Using only the references listed above, answer the following question: \n"
            user_message += f"Current Time: {query_time}\n"
            user_message += f"Question: {query}\n"

            if self.is_server:
                # there is no need to wrap the messages into chat when using the server
                # because we use the chat API: chat.completions.create
                formatted_prompts.append(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ]
                )
            else:
                formatted_prompts.append(
                    self.tokenizer.apply_chat_template(
                        [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message},
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )

        return formatted_prompts

    def format_prompts(self, queries, query_times, batch_retrieval_results=[]):
        """
        Formats queries, corresponding query_times and retrieval results using the chat_template of the model.

        Parameters:
        - queries (List[str]): A list of queries to be formatted into prompts.
        - query_times (List[str]): A list of query_time strings corresponding to each query.
        - batch_retrieval_results (List[str])
        """
        system_prompt = "You are provided with a question and various references. Your task is to answer the question succinctly, using the fewest words possible. If the references do not contain the necessary information to answer the question, respond with 'I don't know'. There is no need to explain the reasoning behind your answers."
        formatted_prompts = []

        for _idx, query in enumerate(queries):
            query_time = query_times[_idx]
            retrieval_results = batch_retrieval_results[_idx]

            user_message = ""
            references = ""

            if len(retrieval_results) > 0:
                references += "# References \n"
                # Format the top sentences as references in the model's prompt template.
                for _snippet_idx, snippet in enumerate(retrieval_results):
                    references += f"- {snippet.strip()}\n"

            references = references[:MAX_CONTEXT_REFERENCES_LENGTH]
            # Limit the length of references to fit the model's input size.

            user_message += f"{references}\n------\n\n"
            user_message += f"Using only the references listed above, answer the following question: \n"
            user_message += f"Current Time: {query_time}\n"
            user_message += f"Question: {query}\n"

            if self.is_server:
                # there is no need to wrap the messages into chat when using the server
                # because we use the chat API: chat.completions.create
                formatted_prompts.append(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ]
                )
            else:
                formatted_prompts.append(
                    self.tokenizer.apply_chat_template(
                        [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message},
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )

        return formatted_prompts



