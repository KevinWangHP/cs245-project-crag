#!/bin/bash

#datasets=("asqa" "hotpot-qa" "nq" "trivia-qa" "musique")
datasets=("trivia-qa")
splits=("trainfew")
rewrite_method="slimplmqr"

for dataset in "${datasets[@]}";
do
    for split in "${splits[@]}";
        do
            python html4rag/bing_url_request.py \
                --dataset=${dataset} \
                --split=${split} \
                --rewrite_method=${rewrite_method}
    done
done

#  --mini_dataset