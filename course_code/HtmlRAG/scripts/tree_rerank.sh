#!/bin/bash

#datasets=("asqa" "hotpot-qa" "nq" "trivia-qa" "musique" "eli5")
datasets=("asqa" "hotpot-qa" "nq" "trivia-qa" "musique" "eli5")
split="test"
rerank_model="bgelargeen"
#rerank_model="e5-mistral"
rewrite_method="slimplmqr"
#url="http://172.16.19.233"
url="http://172.16.20.131"
#url="http://172.16.22.249"

for dataset in "${datasets[@]}";
do
  python html4rag/tree_rerank.py \
    --rerank_model=${rerank_model} \
    --rewrite_method=${rewrite_method} \
    --dataset=${dataset} \
    --url=${url} \
    --max_node_words 0 \
    --split=${split}
done

#  --mini_dataset \