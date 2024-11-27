import argparse
import concurrent.futures
import json
import logging
import os
import re
import threading
import time

import loguru
import requests
from requests.adapters import HTTPAdapter
import random
from tqdm import tqdm
from bs4 import BeautifulSoup as bs, Comment
from urllib3 import Retry

# url = "http://pai-console.c95ec79dd6bf04b63a3403cac3ae3617d.cn-wulanchabu.alicontainer.com/index?workspaceId=ws11231tya14xbmu#/eas/services/bc_search_longctx_exp2/serviceDetails"
# curl http://bc-search-chat-online.gw-gqqd25no78ncp72xfw-1151584402193309.cn-wulanchabu.pai-eas.aliyuncs.com -X POST -d '{"inputs":"<C_Q>你好<C_A>","parameters":{"repetition_penalty":1.05,"temperature":0.3,"top_k":5,"top_p":0.85,"max_new_tokens":2048,"do_sample":true, "seed": 3}}' -H 'Content-Type: application/json'

html_prompt_format_zh = "{pre_text} ```{html_content}``` 请参考以上html网页中的内容，回答问题。{question} {post_text}"
html_prompt_format_en = "{pre_text} ```{html_content}``` Please refer to the above html content and answer the question. {question} {post_text}"
raw_text_prompt_format_zh = "{pre_text} ```{raw_text_content}``` 请参考以上文本内容，回答问题。{question} {post_text}"
raw_text_prompt_format_en = "{pre_text} ```{raw_text_content}``` Please refer to the above text content and answer the question. {question} {post_text}"
markdown_prompt_format_zh = "{pre_text} ```{markdown_content}``` 请参考以上markdown内容，回答问题。{question} {post_text}"
markdown_prompt_format_en = "{pre_text} ```{markdown_content}``` Please refer to the above markdown content and answer the question. {question} {post_text}"

headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("h3", "Header 3"),
    ("h4", "Header 4"),
    ("body", "Body"),
]


def bs_parse(html_content):
    soup = bs(html_content, 'html.parser')
    texts = soup.find_all(string=True)
    texts = [((i.parent.name if i.parent.name else ""), i.text.strip()) for i in texts if i.text.strip()]
    title = ""
    for i in texts:
        if i[0] in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            title = i[1]
            break
    if title == "":
        for i in texts:
            if i[0] == 'title':
                title = i[1]
                break
    texts = [i[1] for i in texts]
    texts = " ".join(texts)
    return texts


def html2raw_text(html_content):
    if isinstance(html_content, list):
        raw_text = [bs_parse(i) for i in html_content]
        raw_text = " ".join(raw_text)
    else:
        raw_text = bs_parse(html_content)
    return raw_text


def html2markdown(html_content):
    from markdownify import markdownify as md
    if isinstance(html_content, list):
        markdown = []
        for i in range(len(html_content)):
            try:
                markdown.append(md(html_content[i]))
            except Exception as e:
                loguru.logger.info(f"convert html to markdown failed: {str(e)}")
                loguru.logger.info("convert to raw text instead")
                markdown.append(bs_parse(html_content[i]))

        markdown = " ".join(markdown)

    else:
        try:
            markdown = md(html_content)
        except Exception as e:
            loguru.logger.info(f"convert html to markdown failed: {str(e)}")
            loguru.logger.info("convert to raw text instead")
            markdown = bs_parse(html_content)
    return markdown


def tgi_api_call(url, prompt, repetition_penalty=-1, temperature=-1, top_k=-1, top_p=-1,
                 max_new_tokens=-1, do_sample=True, seed=None):
    retry_strategy = Retry(
        total=1,  # 最大重试次数（包括首次请求）
        backoff_factor=1,  # 重试之间的等待时间因子
        status_forcelist=[429, 500, 502, 503, 504],  # 需要重试的状态码列表
        allowed_methods=["POST"]  # 只对POST请求进行重试
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    # 创建会话并添加重试逻辑
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    request_data = {
        'inputs': prompt,
        "parameters": {
            "repetition_penalty": repetition_penalty,
            "temperature": temperature,
            "max_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
    }
    if seed != None:
        request_data["parameters"]['seed'] = seed
    if top_p != -1 and top_p != None:
        request_data["parameters"]["top_p"] = top_p
    if top_k != -1 and top_k != None:
        request_data["parameters"]["top_k"] = top_k
    try:
        response = session.post(url, json=request_data, stream=True)
        data = json.loads(response.text)
        if isinstance(data, list):
            ans = data[0]["generated_text"]
        else:
            ans = data["generated_text"]
    except Exception as e:
        loguru.logger.error(response.text)
        # loguru.logger.info(str(e))
        ans = response.text
    return ans


def vllm_api_call(url, prompt, repetition_penalty=-1, temperature=-1, top_k=-1, top_p=-1,
                  max_new_tokens=-1, do_sample=True, seed=None):
    retry_strategy = Retry(
        total=1,  # 最大重试次数（包括首次请求）
        backoff_factor=1,  # 重试之间的等待时间因子
        status_forcelist=[429, 500, 502, 503, 504],  # 需要重试的状态码列表
        allowed_methods=["POST"]  # 只对POST请求进行重试
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    # 创建会话并添加重试逻辑
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    request_data = {
        'prompt': prompt,
        "repetition_penalty": repetition_penalty,
        "temperature": temperature,
        "max_tokens": max_new_tokens,
    }
    if seed != None:
        request_data['seed'] = seed
    if top_p != -1 and top_p != None:
        request_data["top_p"] = top_p
    if top_k != -1 and top_k != None:
        request_data["top_k"] = top_k
    try:
        response = session.post(url, json=request_data)
        data = json.loads(response.text)
        ans = data["text"][0][len(prompt):].strip()
    except Exception as e:
        loguru.logger.error(response.text)
        # loguru.logger.info(str(e))
        ans = response.text
    return ans


def chat_inference(model_input, url):
    # with open("input.txt", "w") as f:
    #     f.write(model_input)
    if chat_model in ["moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k"]:
        return moonshot.predict(text=model_input, model=chat_model, system_prompt=KIMI_SYSTEM, temperature=0.3,
                                top_p=0.85, verbose=False)
    elif chat_model in ["claude-3-opus-20240229"]:
        return claude.predict(text=model_input, verbose=False)
    else:
        if api_type == "tgi":
            return tgi_api_call(url, model_input, repetition_penalty=1.05, temperature=0.3, top_k=5, top_p=0.85,
                                max_new_tokens=1024, do_sample=True, seed=3)
        elif api_type == "vllm":
            return vllm_api_call(url, model_input, repetition_penalty=1.05, temperature=0.3, top_k=5, top_p=0.85,
                                 max_new_tokens=1024, do_sample=True, seed=3)
        else:
            raise NotImplementedError(f"api_type {api_type} not implemented")


def configure_max_context_window(chat_model):
    if re.match(r"bc\d+b\d+k", chat_model):
        max_context_window = re.match(r"bc\d+b(\d+)k", chat_model).group(1)
        max_context_window = int(max_context_window) * 1000
    elif re.match(r"qwen\d+b\d+k", chat_model):
        max_context_window = re.match(r"qwen\d+b(\d+)k", chat_model).group(1)
        max_context_window = int(max_context_window) * 1000
    elif re.match(r"llama\d+b\d+k", chat_model):
        max_context_window = re.match(r"llama\d+b(\d+)k", chat_model).group(1)
        max_context_window = int(max_context_window) * 1000
    elif re.match(r"moonshot-v1-\d+k", chat_model):
        max_context_window = re.match(r"moonshot-v1-(\d+)k", chat_model).group(1)
        max_context_window = int(max_context_window) * 1000
    elif chat_model in ["claude-3-opus-20240229"]:
        max_context_window = 190000
    else:
        max_context_window = 32000
    return max_context_window


def truncate_input(html, max_context_window=32000):
    if isinstance(html, list):
        html = " ".join(html)
    #  if html is longer than 30000 tokens, truncate it
    tokens = tokenizer.tokenize(html)
    if len(tokens) > max_context_window:
        html = tokenizer.convert_tokens_to_string(tokens[:max_context_window])
        # loguru.logger.info(f"html truncated to {max_context_window} tokens")
    return html


from io import StringIO


def convert_possible_tags_to_header(html_content: str) -> str:
    try:
        from lxml import etree
    except ImportError as e:
        raise ImportError(
            "Unable to import lxml, please install with `pip install lxml`."
        ) from e
    # use lxml library to parse html document and return xml ElementTree
    parser = etree.HTMLParser()
    tree = etree.parse(StringIO(html_content), parser)

    xslt_tree = etree.parse("./html4rag/converting_to_header.xslt")
    transform = etree.XSLT(xslt_tree)
    result = transform(tree)
    return str(result)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--chat_model", type=str, default="bc33b192k")
    argparser.add_argument("--dataset", type=str, default="wstqa")
    argparser.add_argument("--split", type=str, default="test")
    argparser.add_argument("--url", type=str,
                           default="http://pai-console.c95ec79dd6bf04b63a3403cac3ae3617d.cn-wulanchabu.alicontainer.com/index?workspaceId=ws11231tya14xbmu#/eas/services/bc_search_longctx_exp2/serviceDetails")
    argparser.add_argument("--reference_format", type=str, default="html")
    argparser.add_argument("--multi_docs", type=str, default="single")
    argparser.add_argument("--multi_qas", action="store_true")
    argparser.add_argument("--mini_dataset", action="store_true")
    argparser.add_argument("--rewrite_method", type=str, default="slimplmqr")
    argparser.add_argument("--rerank_model", type=str, default="bgelargeen")
    argparser.add_argument("--version", type=str, default="v0712")
    argparser.add_argument("--api_type", type=str, default="vllm")
    argparser.add_argument("--offline", action="store_true")
    argparser.add_argument("--src_granularity", type=int, default=256)
    argparser.add_argument("--granularity", type=int, default=128)
    args = argparser.parse_args()
    chat_model = args.chat_model
    dataset = args.dataset
    split = args.split
    url = args.url
    reference_format = args.reference_format
    multi_docs = args.multi_docs
    rewrite_method = args.rewrite_method
    rerank_model = args.rerank_model
    version = args.version
    api_type = args.api_type
    src_granularity = args.src_granularity
    granularity = args.granularity
    search_engine = "bing"
    max_context_window = configure_max_context_window(chat_model)
    if reference_format in ["html-trim", "fill-chunk", "tree-gen", "chunk-rerank-tree-gen"]:
        multi_docs = "chunk"
        loguru.logger.info(f"changing multi_docs to chunk for reference_format {reference_format}")
        loguru.logger.info(f"rewrite_method: {rewrite_method}, rerank_model: {rerank_model}")

    if dataset in ["wstqa", "statsgov"]:
        language = "zh"
        loguru.logger.info(f"setting language to {language} for dataset {dataset}")
    elif dataset in ["websrc", "asqa", "nq", "hotpot-qa", "trivia-qa", "eli5", "musique"]:
        language = "en"
        loguru.logger.info(f"setting language to {language} for dataset {dataset}")
    else:
        raise NotImplementedError(f"dataset {dataset} not implemented")

    if re.match(r"bc\d+b\d+k", chat_model) or re.match(r"qwen\d+b\d+k", chat_model):
        pre_text = "<C_Q>"
        post_text = "<C_A>"
        tokenizer_path = "../../huggingface/Baichuan2-7B-Chat/"
        tokenizer_name = "bc"
    if re.match(r"llama\d+b\d+k", chat_model):
        pre_text = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        post_text = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        tokenizer_path = "../../huggingface/Meta-Llama-3.1-70B-Instruct/"
        tokenizer_name = "llama"
    elif chat_model in ["mistral7b"]:
        pre_text = "<s>[INST]"
        post_text = "[/INST]"
        tokenizer_path = "../../huggingface/Mistral-7B-Instruct-v0.2/"
        tokenizer_name = "mistral"
    elif chat_model in ["qwen32b"]:
        pre_text = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
        post_text = "<|im_end|>\n<|im_start|>assistant\n"
        tokenizer_path = "../../huggingface/Qwen1.5-32B-Chat"
        tokenizer_name = "qwen"
    elif chat_model in ["moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k"]:
        from moonshot import Moonshot, KIMI_SYSTEM
        moonshot = Moonshot()
        pre_text = ""
        post_text = ""
        tokenizer_path = "../../huggingface/Baichuan2-7B-Chat/"
    elif chat_model in ["claude-3-opus-20240229"]:
        from claude import ClaudeProxy
        claude = ClaudeProxy()
        pre_text = ""
        post_text = ""
        tokenizer_path = "../../huggingface/Baichuan2-7B-Chat/"
    else:
        raise NotImplementedError(f"chat_model {chat_model} not implemented")

    context_window = f"{max_context_window // 1000}k"
    if multi_docs != "single":
        if reference_format == "html-simple":
            loguru.logger.info(f"changing input_file to simple format for reference_format {reference_format}")
            input_file = f"./html_data/{dataset}/{search_engine}/{search_engine}html-{rewrite_method}-{dataset}-simple-{split}.jsonl"
        elif reference_format == "html-attr":
            input_file = f"./html_data/{dataset}/{search_engine}/{search_engine}html-{rewrite_method}-{dataset}-simplewithattr-{split}.jsonl"
        elif reference_format in ["tree-rerank"]:
            input_file = f"./html_data/{dataset}/{reference_format}/{tokenizer_name}/{search_engine}html-{rewrite_method}-{rerank_model}-{granularity}-{dataset}-{split}-{context_window}.jsonl"
        elif reference_format in ["html-trim", "fill-chunk"]:
            #  read file according to context window
            loguru.logger.info(f"changing input_file to trim format for reference_format {reference_format}")
            input_file = f"./html_data/{dataset}/{reference_format}/{tokenizer_name}/{search_engine}html-{rewrite_method}-{rerank_model}-{dataset}-{split}-{context_window}.jsonl"
        elif reference_format == "tree-gen":
            input_file = f"./html_data/{dataset}/tree-gen/{version}/{tokenizer_name}/{search_engine}html-{rewrite_method}-{version}-{granularity}-{dataset}-{split}-{context_window}.jsonl"
        elif reference_format in ["chunk-rerank-tree-gen", "tree-rerank-tree-gen"]:
            if dataset in ["asqa", "nq", "eli5"]:
                coarse_context_window = {"1k": "2k", "2k": "3k", "4k": "6k", "8k": "12k", "16k": "24k", "32k": "48k", "64k": "96k"}[context_window]
            else:
                coarse_context_window = {"1k": "2k", "2k": "4k", "4k": "8k", "8k": "16k", "16k": "24k", "32k": "48k", "64k": "96k"}[context_window]
            input_file = f"./html_data/{dataset}/{reference_format}/{version}/{tokenizer_name}/{search_engine}html-{rewrite_method}-{rerank_model}-{src_granularity}to{granularity}-{dataset}-{split}-{coarse_context_window}to{context_window}.jsonl"
        elif reference_format in ["llmlingua", "bgelargeen", "jinaai-reader", "e5-mistral", "bm25"]:
            input_file = f"./html_data/{dataset}/baselines/{tokenizer_name}/{search_engine}html-{rewrite_method}-{reference_format}-{dataset}-{split}-{context_window}.jsonl"
        else:
            # multi docs
            input_file = f"./html_data/{dataset}/{search_engine}/{search_engine}html-{rewrite_method}-{dataset}-{split}.jsonl"
    else:
        input_file = f"./html_data/{dataset}/{dataset}-{split}.jsonl"

    loguru.logger.info(f"input_file: {input_file}")
    data_lines = [json.loads(l) for l in open(input_file)]

    if args.mini_dataset:
        data_lines = data_lines[:10]

    if multi_docs == "single":
        output_dir = f"./html_data/{dataset}/{chat_model}"
    else:
        output_dir = f"./html_data/{dataset}/{chat_model}/{search_engine}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if multi_docs == "single":
        output_file = f"{output_dir}/{chat_model}-{reference_format}-{dataset}-{split}.jsonl"
    elif reference_format in ["html-trim", "fill-chunk"]:
        output_file = f"{output_dir}/{chat_model}-{reference_format}-{rewrite_method}-{rerank_model}-{dataset}-{split}.jsonl"
    elif reference_format == "tree-gen":
        output_file = f"{output_dir}/{chat_model}-{reference_format}-{rewrite_method}-{version}-{granularity}-{dataset}-{split}.jsonl"
    elif reference_format == "tree-rerank":
        output_file = f"{output_dir}/{chat_model}-{reference_format}-{rewrite_method}-{rerank_model}-{granularity}-{dataset}-{split}.jsonl"
    elif reference_format in ["chunk-rerank-tree-gen", "tree-rerank-tree-gen"]:
        output_file = f"{output_dir}/{chat_model}-{reference_format}-{rewrite_method}-{rerank_model}-{src_granularity}to{granularity}-{coarse_context_window}-{version}-{dataset}-{split}.jsonl"
    else:
        output_file = f"{output_dir}/{chat_model}-{reference_format}-{rewrite_method}-{dataset}-{split}.jsonl"
    with open(output_file, "w") as f:
        pass



    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    loguru.logger.info(f"url: {url}")
    loguru.logger.info(f"dataset: {dataset}, split: {split}, chat_model: {chat_model}, reference_format: {reference_format}")

    model_inputs = []
    numthreads = min(32, 256 // (max_context_window // 1000))
    loguru.logger.info(f"numthreads: {numthreads}")
    for idx in tqdm(range(len(data_lines)), desc=f"{dataset}-{split} prepare input", total=len(data_lines)):

        if reference_format in ["html-trim", "tree-gen", "chunk-rerank-tree-gen", "tree-rerank", "tree-rerank-tree-gen"]:
            #  trim html
            if isinstance(data_lines[idx]['html_trim'], list):
                html = " ".join(data_lines[idx]["html_trim"])
            else:
                html = data_lines[idx]["html_trim"]
            if language == "en":
                model_input = html_prompt_format_en.format(pre_text=pre_text, post_text=post_text,
                                                           html_content=html,
                                                           question=data_lines[idx]['question'])
            elif language == "zh":
                model_input = html_prompt_format_zh.format(pre_text=pre_text, post_text=post_text,
                                                           html_content=html,
                                                           question=data_lines[idx]['question'])
        elif reference_format in ["llmlingua", "fill-chunk", "bgelargeen", "e5-mistral", "bm25"]:
            if isinstance(data_lines[idx]['html_trim'], list):
                ref = " ".join(data_lines[idx]["html_trim"])
            else:
                ref = data_lines[idx]["html_trim"]
            markdown_text = truncate_input(ref, max_context_window=max_context_window)
            if language == "en":
                model_input = raw_text_prompt_format_en.format(pre_text=pre_text, post_text=post_text,
                                                               raw_text_content=markdown_text,
                                                               question=data_lines[idx]['question'])
            elif language == "zh":
                model_input = raw_text_prompt_format_zh.format(pre_text=pre_text, post_text=post_text,
                                                               raw_text_content=markdown_text,
                                                               question=data_lines[idx]['question'])
        elif reference_format in ["jinaai-reader"]:
            if isinstance(data_lines[idx]['html_trim'], list):
                ref = " ".join(data_lines[idx]["html_trim"])
            else:
                ref = data_lines[idx]["html_trim"]
            markdown_text = truncate_input(ref, max_context_window=max_context_window)
            if language == "en":
                model_input = markdown_prompt_format_en.format(pre_text=pre_text, post_text=post_text,
                                                               markdown_content=markdown_text,
                                                               question=data_lines[idx]['question'])
            elif language == "zh":
                model_input = markdown_prompt_format_zh.format(pre_text=pre_text, post_text=post_text,
                                                               markdown_content=markdown_text,
                                                               question=data_lines[idx]['question'])
        else:
            if re.match(r"top\d+", multi_docs):
                top_n = int(multi_docs[3:])
                html = [d['html'] for d in data_lines[idx][f'{rewrite_method}_results'] if d['html']]
                if len(html) > top_n:
                    html = html[:top_n]
            elif multi_docs == "single":
                html = data_lines[idx]['html']
            elif multi_docs == "chunk":
                pass
            else:
                raise NotImplementedError(f"multi_docs {multi_docs} not implemented")

            if reference_format in ["html", "html-simple", "html-attr"]:
                if language == "en":
                    model_input = html_prompt_format_en.format(pre_text=pre_text, post_text=post_text,
                                                               html_content=truncate_input(html,
                                                                                           max_context_window=max_context_window),
                                                               question=data_lines[idx]['question'])
                elif language == "zh":
                    model_input = html_prompt_format_zh.format(pre_text=pre_text, post_text=post_text,
                                                               html_content=truncate_input(html,
                                                                                           max_context_window=max_context_window),
                                                               question=data_lines[idx]['question'])
            elif reference_format == "raw-text":
                if language == "en":
                    model_input = raw_text_prompt_format_en.format(pre_text=pre_text, post_text=post_text,
                                                                   raw_text_content=truncate_input(html2raw_text(html),
                                                                                                   max_context_window=max_context_window),
                                                                   question=data_lines[idx]['question'])
                elif language == "zh":
                    model_input = raw_text_prompt_format_zh.format(pre_text=pre_text, post_text=post_text,
                                                                   raw_text_content=truncate_input(html2raw_text(html),
                                                                                                   max_context_window=max_context_window),
                                                                   question=data_lines[idx]['question'])
            elif reference_format == "markdown":
                if language == "en":
                    model_input = markdown_prompt_format_en.format(pre_text=pre_text, post_text=post_text,
                                                                   markdown_content=truncate_input(html2markdown(html),
                                                                                                   max_context_window=max_context_window),
                                                                   question=data_lines[idx]['question'])
                elif language == "zh":
                    model_input = markdown_prompt_format_zh.format(pre_text=pre_text, post_text=post_text,
                                                                   markdown_content=truncate_input(html2markdown(html),
                                                                                                   max_context_window=max_context_window),
                                                                   question=data_lines[idx]['question'])
            else:
                raise NotImplementedError(f"reference_format {reference_format} not implemented")
        model_inputs.append(model_input)
    threads = []


    def get_output(model_input, idx):
        while True:
            res = chat_inference(model_input, url)
            if "stream timeout" in res:
                # loguru.logger.info(f"stream timeout, retrying...")
                time.sleep(random.random() * 5 + 3)
            else:
                data_lines[idx][f"{chat_model}_{reference_format}"] = res
                break


    if args.offline:
        from vllm import SamplingParams, LLM
        import torch
        sampling_params = SamplingParams(
            repetition_penalty=1.05,
            temperature=0.3,
            top_k=5,
            top_p=0.85,
            seed=3
        )
        # Create an LLM.
        ngpus = torch.cuda.device_count()
        # llm = LLM(model=tokenizer_path, trust_remote_code=True, dtype="bfloat16", tensor_parallel_size=ngpus)
        llm = LLM(model=tokenizer_path, tensor_parallel_size=ngpus)
        # Generate texts from the prompts. The output is a list of RequestOutput objects
        # that contain the prompt, generated text, and other information.
        outputs = llm.generate(model_inputs, sampling_params)
        # loguru.logger.info the outputs.
        oidx = 0
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            data_lines[oidx][f"{chat_model}_{reference_format}"] = generated_text
            oidx += 1
    else:
        for idx, data_point in tqdm(enumerate(data_lines), desc=f"{dataset}-{split} chat inference",
                                    total=len(data_lines)):
            thread = threading.Thread(
                target=get_output,
                args=(model_inputs[idx], idx)
            )
            thread.start()
            threads.append(thread)
            if len(threads) == numthreads:
                for thread in threads:
                    thread.join()
                threads = []

        # collect the final threads
        if len(data_lines) > 0:
            for thread in threads:
                thread.join()

    # with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
    #     for idx in tqdm(range(len(data_lines)), desc=f"{dataset}-{split}", total=len(data_lines)):
    #         # if idx == 0:
    #         #     loguru.logger.info(f"model_input: \n{model_input}")
    #
    #         future = executor.submit(chat_inference, model_inputs[idx], url)
    #         response = future.result()
    #         if post_text in response and not response.endswith(post_text):
    #             response = response.split(post_text)[1]
    #         data_lines[idx][f"{chat_model}_{reference_format}"] = response

    with open(output_file, "a") as f:
        for data_line in data_lines:
            f.write(json.dumps(data_line, ensure_ascii=False) + "\n")
    loguru.logger.info(f"output_file: {output_file}")
