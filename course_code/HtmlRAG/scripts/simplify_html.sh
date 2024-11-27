#!/bin/bash

split="test"
rewrite_method="slimplmqr"

python html4rag/simplify_html.py \
    --split=${split} \
    --keep_attr \
    --rewrite_method=${rewrite_method}


#    --mini_dataset