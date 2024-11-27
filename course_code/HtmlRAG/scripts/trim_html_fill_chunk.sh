#!/bin/bash

split="test"
#rerank_model="bgelargeen"
#rerank_model="e5-mistral"
#rerank_model="bm25"
rewrite_method="slimplmqr"
chat_tokenizer_name="llama"

python html4rag/trim_html_fill_chunk.py \
    --split=${split} \
    --rerank_model=${rerank_model} \
    --rewrite_method=${rewrite_method} \
    --chat_tokenizer_name=${chat_tokenizer_name}
#  --mini_dataset \