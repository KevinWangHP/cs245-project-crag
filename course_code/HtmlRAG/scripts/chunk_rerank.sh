#!/bin/bash

datasets=("asqa" "hotpot-qa" "nq" "trivia-qa" "musique" "eli5")
#datasets=("musique" "eli5")
split="test"
#rerank_model="bgelargeen"
#rerank_model="e5-mistral"
#rerank_model="bm25"
rewrite_method="slimplmqr"
url="http://172.16.23.46/"

for dataset in "${datasets[@]}";
do
  python html4rag/chunk_rerank.py \
    --rerank_model=${rerank_model} \
    --rewrite_method=${rewrite_method} \
    --dataset=${dataset} \
    --url=${url} \
    --split=${split}
done

#  --mini_dataset \