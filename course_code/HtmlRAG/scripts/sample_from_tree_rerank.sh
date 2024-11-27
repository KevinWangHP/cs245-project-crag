#!/bin/bash

split="trainfew"
version="v0915"
rewrite_method="slimplmqr"

python html4rag/sample_from_tree_rerank.py \
    --version ${version} \
    --split=${split} \
    --rewrite_method=${rewrite_method}

#    --mini_dataset