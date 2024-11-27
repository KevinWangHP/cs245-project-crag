# pip install transformers
import argparse
import json
import os
import re

import loguru
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

max_pack_length = 64000

# (REMOVE <SCRIPT> to </script> and variations)
SCRIPT_PATTERN = r'<[ ]*script.*?\/[ ]*script[ ]*>'  # mach any char zero or more times
# text = re.sub(pattern, '', text, flags=(re.IGNORECASE | re.MULTILINE | re.DOTALL))

# (REMOVE HTML <STYLE> to </style> and variations)
STYLE_PATTERN = r'<[ ]*style.*?\/[ ]*style[ ]*>'  # mach any char zero or more times
# text = re.sub(pattern, '', text, flags=(re.IGNORECASE | re.MULTILINE | re.DOTALL))

# (REMOVE HTML <META> to </meta> and variations)
META_PATTERN = r'<[ ]*meta.*?>'  # mach any char zero or more times
# text = re.sub(pattern, '', text, flags=(re.IGNORECASE | re.MULTILINE | re.DOTALL))

# (REMOVE HTML COMMENTS <!-- to --> and variations)
COMMENT_PATTERN = r'<[ ]*!--.*?--[ ]*>'  # mach any char zero or more times
# text = re.sub(pattern, '', text, flags=(re.IGNORECASE | re.MULTILINE | re.DOTALL))

# (REMOVE HTML LINK <LINK> to </link> and variations)
LINK_PATTERN = r'<[ ]*link.*?>'  # mach any char zero or more times

# (REPLACE base64 images)
BASE64_IMG_PATTERN = r'<img[^>]+src="data:image/[^;]+;base64,[^"]+"[^>]*>'

# (REPLACE <svg> to </svg> and variations)
SVG_PATTERN = r'(<svg[^>]*>)(.*?)(<\/svg>)'

def create_prompt(text:str, tokenizer) -> str:
   messages = [
    {
        "role": "user",
        "content": text
    },
   ]
   return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
   )

def replace_svg(html: str, new_content: str = "this is a placeholder") -> str:
    return re.sub(
        SVG_PATTERN,
        lambda match: f"{match.group(1)}{new_content}{match.group(3)}",
        html,
        flags=re.DOTALL,
    )

def replace_base64_images(html: str, new_image_src: str = "#") -> str:
    return re.sub(BASE64_IMG_PATTERN, f'<img src="{new_image_src}"/>', html)

def clean_html(html: str, clean_svg: bool = False, clean_base64: bool = False):
    html = re.sub(SCRIPT_PATTERN, '', html, flags=(re.IGNORECASE | re.MULTILINE | re.DOTALL))
    html = re.sub(STYLE_PATTERN, '', html, flags=(re.IGNORECASE | re.MULTILINE | re.DOTALL))
    html = re.sub(META_PATTERN, '', html, flags=(re.IGNORECASE | re.MULTILINE | re.DOTALL))
    html = re.sub(COMMENT_PATTERN, '', html, flags=(re.IGNORECASE | re.MULTILINE | re.DOTALL))
    html = re.sub(LINK_PATTERN, '', html, flags=(re.IGNORECASE | re.MULTILINE | re.DOTALL))

    if clean_svg:
        html = replace_svg(html)

    if clean_base64:
        html = replace_base64_images(html)

    return html

def pack_htmls(htmls, tokenizer, num_packs):
    packs=[[] for _ in range(num_packs)]
    pack_lengths=[0 for _ in range(num_packs)]
    num_tokens=[len(tokenizer.encode(h, add_special_tokens=False)) for h in htmls]
    for i, h in enumerate(htmls):
        #. get the pack with the smallest length
        min_pack_length=min(pack_lengths)
        if min_pack_length>max_pack_length:
            break
        pack_idx=pack_lengths.index(min_pack_length)
        packs[pack_idx].append(h)
        pack_lengths[pack_idx]+=num_tokens[i]
    return packs


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--chat_model", type=str, default="bc33b192k")
    argparser.add_argument("--dataset", type=str, default="wstqa")
    argparser.add_argument("--split", type=str, default="test")
    argparser.add_argument("--search_engine", type=str, default="bing")
    argparser.add_argument("--mini_dataset", action="store_true")
    argparser.add_argument("--rewrite_method", type=str, default="slimplmqr")
    argparser.add_argument("--refiner_name", type=str, default="llmlingua")
    argparser.add_argument("--context_window", type=str, default="2k")
    argparser.add_argument("--chat_tokenizer_name", type=str, default="llama")
    args = argparser.parse_args()

    dataset = args.dataset
    split = args.split
    search_engine = args.search_engine
    mini_dataset = args.mini_dataset
    rewrite_method = args.rewrite_method
    refiner_name = args.refiner_name
    context_window = args.context_window
    chat_tokenizer_name = args.chat_tokenizer_name

    # max_context_window = int(context_window[:-1]) * 1000
    num_packs=int(context_window[:-1])
    loguru.logger.info(f"split into {num_packs} packs")
    checkpoint = "../../huggingface/reader-lm-1.5b"


    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)

    if rewrite_method in ["slimplmqr", ]:
        input_file = f"./html_data/{dataset}/{search_engine}/{search_engine}html-{rewrite_method}-{dataset}-simple-{split}.jsonl"
        output_file = f"./html_data/{dataset}/baselines/{chat_tokenizer_name}/{search_engine}html-{rewrite_method}-{refiner_name}-{dataset}-{split}-{context_window}.jsonl"
    else:
        raise NotImplementedError(f"rewrite method {rewrite_method} not implemented")
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))

    loguru.logger.info(f"input_file: {input_file}")
    data_lines = [json.loads(l) for l in open(input_file)]
    if args.mini_dataset:
        data_lines = data_lines[:10]


    # init refiners on different devices
    refiner_pool = []
    thread_pool = []

    from vllm import SamplingParams

    top_k = 1  # @param {type:"integer"}
    temperature = 0  # @param {type:"slider", min:0, max:1, step:0.1}
    repetition_penalty = 1.08  # @param {type:"number"}
    presence_penalty = 0.25  # @param {type:"slider", min:0, max:1, step:0.1}
    top_k = 1  # @param {type:"integer"}
    max_tokens = 1000  # @param {type:"integer"}

    sampling_params = SamplingParams(temperature=temperature, top_k=top_k, presence_penalty=presence_penalty,
                                     repetition_penalty=repetition_penalty, max_tokens=max_tokens)

    print('sampling_params', sampling_params)

    from vllm import LLM
    llm = LLM(model=checkpoint, dtype='float16', trust_remote_code=True, tensor_parallel_size=2, gpu_memory_utilization=0.98)
    prompts=[]
    for i, data_line in enumerate(tqdm(data_lines, desc="prepare data")):
        htmls=[h["html"] for h in data_line[f"{rewrite_method}_results"]]
        html_packs = pack_htmls(htmls, tokenizer, num_packs)
        html_trim=[]
        for pack in html_packs:
            html_str="\n".join(pack)
            html_str = clean_html(html_str, clean_svg=True, clean_base64=True)
            input_tokens=tokenizer.encode(html_str, add_special_tokens=False)
            if len(input_tokens)>max_pack_length:
                input_tokens=input_tokens[:max_pack_length]
                html_str=tokenizer.decode(input_tokens)
            prompt = create_prompt(html_str, llm.get_tokenizer())
            prompts.append(prompt)

    results = llm.generate(prompts, sampling_params=sampling_params)
    for i in range(len(data_lines)):
        pack_res=results[i*num_packs:(i+1)*num_packs]
        data_lines[i]["html_trim"] = [str(r.outputs[0].text) for r in pack_res]

    with open(output_file, "w") as f:
        for line in data_lines:
            f.write(json.dumps(line) + "\n")

    loguru.logger.info(f"output_file: {output_file}")

    from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
    import gc
    import os
    import torch

    destroy_model_parallel()
    destroy_distributed_environment()
    del llm.llm_engine.model_executor.driver_worker
    del llm.llm_engine.model_executor
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    print(f"cuda memory: {torch.cuda.memory_allocated() // 1024 // 1024}MB")

