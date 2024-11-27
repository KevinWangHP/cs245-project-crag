#!/bin/bash

#datasets=("wstqa" "websrc" "asqa" "statsgov")
datasets=("asqa" "hotpot-qa" "nq" "trivia-qa" "musique" "eli5")
#datasets=("asqa")
#reference_formats=("html" "raw-text" "markdown" "html-simple" "bm25" "bgelargeen" "tree-rerank" "tree-rerank-tree-gen" "llmlingua" "jinaai-reader" "e5-mistral")
reference_formats=("tree-rerank-tree-gen")
split="test"
#chat_model="claude-3-opus-20240229"
#chat_models=("bc34b8k" "bc34b16k" "bc34b32k" "bc34b64k" "bc34b128k" "bc34b192k")
#chat_models=("bc34b8k" "bc34b16k")
#chat_models=("qwen72b8k" "qwen72b16k" "qwen72b32k" "qwen72b64k" "qwen72b128k" "qwen72b192k")
#chat_models=("qwen72b192k")
#chat_models=("llama70b2k" "llama70b4k" "llama70b8k" "llama70b16k" "llama70b32k")
chat_models=("llama70b4k")
#chat_models=("llama8b2k" "llama8b4k" "llama8b8k" "llama8b16k" "llama8b32k")
#chat_models=("llama8b128k")
url="http://llama31-70b-vllm.search.cls-3nbemh6i.ml.baichuan-inc.com/generate"
#url="http://llama31-70b-vllm-4gpus.search.cls-3nbemh6i.ml.baichuan-inc.com/generate"
#url="http://llama31-8b-vllm-2gpus.search.cls-3nbemh6i.ml.baichuan-inc.com/generate"
#url="http://llama31-8b-vllm-4gpus.search.cls-3nbemh6i.ml.baichuan-inc.com/generate"
multi_docs="top10"
#multi_docs="single"
rerank_model="bgelargeen"
rewrite_method="slimplmqr"
version="v1008"
#granularity=512
granularities=(128)

for granularity in "${granularities[@]}";
do
    for chat_model in "${chat_models[@]}";
    do
        for dataset in "${datasets[@]}";
        do
            for reference_format in "${reference_formats[@]}";
                do
                    python html4rag/chat_inference.py \
                        --multi_docs=${multi_docs} \
                        --chat_model=${chat_model} \
                        --dataset=${dataset} \
                        --split=${split} \
                        --reference_format=${reference_format} \
                        --url=${url} \
                        --rerank_model=${rerank_model} \
                        --rewrite_method=${rewrite_method} \
                        --src_granularity=256 \
                        --granularity=${granularity} \
                        --version=${version}
            done
        done
    done
done

#  --mini_dataset
#  --multi_docs
#  --multi_qas
