#!/bin/bash

#datasets=("asqa" "hotpot-qa" "nq" "trivia-qa" "musique" "eli5")
datasets=("eli5")
split="test"
search_engine="bing"
rewrite_method="slimplmqr"
refiner_name="jinaai-reader"
#context_windows=("1k" "2k" "4k" "8k" "16k" "32k")
context_windows=("4k")

#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
for dataset in "${datasets[@]}";
do
  for context_window in "${context_windows[@]}";
  do
    python flashrag_refiner/jinaai_reader.py \
      --dataset=${dataset} \
      --split=${split} \
      --search_engine=${search_engine} \
      --rewrite_method=${rewrite_method} \
      --context_window=${context_window} \
      --refiner_name=${refiner_name}
  done
done

# --mini_dataset \
# CUDA_VISIBLE_DEVICES=2,3 nohup bash ./scripts/jinaai_reader.sh > jinaai_reader_23.out &
# CUDA_VISIBLE_DEVICES=4,5 nohup bash ./scripts/jinaai_reader.sh > jinaai_reader_45.out &
# CUDA_VISIBLE_DEVICES=6,7 nohup bash ./scripts/jinaai_reader.sh > jinaai_reader_67.out &
