# 🤖🔍 HtmlRAG

<div align="center">
<a href="https://arxiv.org/abs/2411.02959" target="_blank"><img src=https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv></a>
<a href="https://github.com/plageon/HtmlRAG" target="_blank"><img src="https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white"></a>
<a href="https://modelscope.cn/models/zstanjj/HTML-Pruner-Llama-1B" target="_blank"><img src=https://custom-icon-badges.demolab.com/badge/ModelScope%20Models-624aff?style=flat&logo=modelscope&logoColor=white></a>
<a href="https://github.com/plageon/HtmlRAG/blob/main/toolkit/LICENSE"><img alt="License" src="https://img.shields.io/badge/LICENSE-MIT-green"></a>
<a><img alt="Static Badge" src="https://img.shields.io/badge/made_with-Python-blue"></a>
</div>

A toolkit to apply HtmlRAG in your own RAG systems.

## 📦 Installation

Install the package using pip:
```bash
pip install htmlrag
```
Or install the package from source:
```bash
pip install -e .
```

---

## 📖 User Guide

### 🧹 HTML Cleaning

```python
from htmlrag import clean_html

question = "When was the bellagio in las vegas built?"
html = """
<html>
<head>
<title>When was the bellagio in las vegas built?</title>
</head>
<body>
<p class="class0">The Bellagio is a luxury hotel and casino located on the Las Vegas Strip in Paradise, Nevada. It was built in 1998.</p>
</body>
<div>
<div>
<p>Some other text</p>
<p>Some other text</p>
</div>
</div>
<p class="class1"></p>
<!-- Some comment -->
<script type="text/javascript">
document.write("Hello World!");
</script>
</html>
"""

simplified_html = clean_html(html)
print(simplified_html)

# <html>
# <title>When was the bellagio in las vegas built?</title>
# <p>The Bellagio is a luxury hotel and casino located on the Las Vegas Strip in Paradise, Nevada. It was built in 1998.</p>
# <div>
# <p>Some other text</p>
# <p>Some other text</p>
# </div>
# </html>
```


### 🌲 Build Block Tree

```python
from htmlrag import build_block_tree

block_tree, simplified_html = build_block_tree(simplified_html, max_node_words=10)
for block in block_tree:
    print("Block Content: ", block[0])
    print("Block Path: ", block[1])
    print("Is Leaf: ", block[2])
    print("")

# Block Content:  <title>When was the bellagio in las vegas built?</title>
# Block Path:  ['html', 'title']
# Is Leaf:  True
# 
# Block Content:  <div>
# <p>Some other text</p>
# <p>Some other text</p>
# </div>
# Block Path:  ['html', 'div']
# Is Leaf:  True
# 
# Block Content:  <p>The Bellagio is a luxury hotel and casino located on the Las Vegas Strip in Paradise, Nevada. It was built in 1998.</p>
# Block Path:  ['html', 'p']
# Is Leaf:  True
```

### ✂️ Prune HTML Blocks with Embedding Model

```python
from htmlrag import EmbedHTMLPruner

embed_model="/train_data_load/huggingface/tjj_hf/bge-large-en/"
query_instruction_for_retrieval = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
embed_html_pruner = EmbedHTMLPruner(embed_model=embed_model, local_inference=True, query_instruction_for_retrieval = query_instruction_for_retrieval)
# alternatively you can init a remote TEI model, refer to https://github.com/huggingface/text-embeddings-inference.
# tei_endpoint="http://YOUR_TEI_ENDPOINT"
# embed_html_pruner = EmbedHTMLPruner(embed_model=embed_model, local_inference=False, query_instruction_for_retrieval = query_instruction_for_retrieval, endpoint=tei_endpoint)
block_rankings=embed_html_pruner.calculate_block_rankings(question, simplified_html, block_tree)
print(block_rankings)

# [0, 2, 1]

#. alternatively you can use bm25 to rank the blocks
from htmlrag import BM25HTMLPruner
bm25_html_pruner = BM25HTMLPruner()
block_rankings=bm25_html_pruner.calculate_block_rankings(question, simplified_html, block_tree)
print(block_rankings)

# [0, 2, 1]

from transformers import AutoTokenizer

chat_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-70B-Instruct")

max_context_window = 60
pruned_html = embed_html_pruner.prune_HTML(simplified_html, block_tree, block_rankings, chat_tokenizer, max_context_window)
print(pruned_html)

# <html>
# <title>When was the bellagio in las vegas built?</title>
# <p>The Bellagio is a luxury hotel and casino located on the Las Vegas Strip in Paradise, Nevada. It was built in 1998.</p>
# </html>
```


### ✂️ Prune HTML Blocks with Generative Model

```python
from htmlrag import GenHTMLPruner
import torch

ckpt_path = "zstanjj/HTML-Pruner-Llama-1B"
if torch.cuda.is_available():
    device="cuda"
else:
    device="cpu"
gen_embed_pruner = GenHTMLPruner(gen_model=ckpt_path, max_node_words=5, device=device)
block_rankings = gen_embed_pruner.calculate_block_rankings(question, pruned_html)
print(block_rankings)

# [1, 0]

block_tree, pruned_html=build_block_tree(pruned_html, max_node_words=10)
for block in block_tree:
    print("Block Content: ", block[0])
    print("Block Path: ", block[1])
    print("Is Leaf: ", block[2])
    print("")
    
# Block Content:  <title>When was the bellagio in las vegas built?</title>
# Block Path:  ['html', 'title']
# Is Leaf:  True
# 
# Block Content:  <p>The Bellagio is a luxury hotel and casino located on the Las Vegas Strip in Paradise, Nevada. It was built in 1998.</p>
# Block Path:  ['html', 'p']
# Is Leaf:  True

max_context_window = 32
pruned_html = gen_embed_pruner.prune_HTML(pruned_html, block_tree, block_rankings, chat_tokenizer, max_context_window)
print(pruned_html)

# <p>The Bellagio is a luxury hotel and casino located on the Las Vegas Strip in Paradise, Nevada. It was built in 1998.</p>
```
