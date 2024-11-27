#!/bin/bash

# This script is used to apply search pipeline
#datasets=("asqa" "hotpot-qa" "nq" "trivia-qa" "musique")
datasets=("asqa" )
splits=("test")
search_methods=("vanilla_search")
#search_methods=("slimplmqr")
search_engine="bing"
address="172.17.8.186:5050"

for dataset in "${datasets[@]}";
do
    for split in "${splits[@]}";
    do
        for search_method in "${search_methods[@]}";
        do
            if [ "$search_engine" == "bing" ]; then
                python search_utils/search_pipeline_apply.py \
                    --dataset $dataset \
                    --split $split \
                    --search_method $search_method
            elif [ "$search_engine" == "kiltbm25" ]; then
                 python search_utils/kiltbm25.py \
                      --address $address \
                      --dataset $dataset \
                      --split $split \
                      --search_method $search_method \
                      --search_engine $search_engine
            fi
        done
    done
done
#  mini_dataset
