#!/bin/bash

split="test"
rewrite_method="slimplmqr"
search_engine="bing"

python html4rag/trim_html_tree_rerank.py \
  --split=${split} \
  --rewrite_method=${rewrite_method} \
  --search_engine=${search_engine} \
  --rerank_model="bgelargeen" \
  --max_node_words 0 \
  --chat_tokenizer_name "llama"

#chown -R bc_search_intern ./html_data/

#  --mini_dataset \