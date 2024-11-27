#!/bin/bash

search_engine="bing"

split="test"
rewrite_method="slimplmqr"
#ckpt_path="../../model/glm4-9b/glm4-9b-256k-v0715-node16-acc4/checkpoint-1060"
#ckpt_path="../../model/glm4-9b/glm4-9b-128k-v0701-node8-gacc4/checkpoint-770"
#ckpt_path="../../model/train-tree-rerank-phi35-mini/v0907/checkpoint-340"
#ckpt_path="../../model/train-tree-rerank-phi35-mini/v0908/checkpoint-344"
#ckpt_path="../../model/train-tree-rerank-phi35-mini/v0914/checkpoint-172"
#ckpt_path="../../model/train-tree-rerank-phi35-mini/v0915/checkpoint-164"
ckpt_path="../../model/train-tree-rerank-llama32/v1008/checkpoint-381"
ckpt_version="v1008"
#base_path="./llm_modeling/ChatGLM"
#base_path="./llm_modeling/Phi35"
base_path="./llm_modeling/Llama32"
rerank_model="bgelargeen"

chat_tokenizer_name="llama"

datasets=("asqa" "hotpot-qa" "nq" "trivia-qa" "musique" "eli5")
#datasets=("asqa" "hotpot-qa" "nq" "trivia-qa" "musique")
#context_windows=("1k" "2k" "4k" "8k" "16k")
context_windows=("4k")
fine_trim_ratios=("custom")

# copy model config file to ckpt_path
#cp ${base_path}/* ${ckpt_path}
ls -al ${ckpt_path}

for context_window in "${context_windows[@]}";
do
  for dataset in "${datasets[@]}";
  do
    for fine_trim_ratio in "${fine_trim_ratios[@]}";
    do
      python html4rag/tree_rerank_tree_gen.py \
        --dataset=${dataset} \
        --split=${split} \
        --rewrite_method=${rewrite_method} \
        --search_engine=${search_engine} \
        --ckpt_version=${ckpt_version} \
        --ckpt_path ${ckpt_path} \
        --context_window ${context_window} \
        --fine_trim_ratio ${fine_trim_ratio} \
        --rerank_model=${rerank_model} \
        --chat_tokenizer_name=${chat_tokenizer_name} \
        --src_max_node_words 256 \
        --max_node_words 128
    done
  done
done

# --mini_dataset \

