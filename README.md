# ColdRAG

This is the official implementation of ColdRAG, a Retrieval-Augmented, Large Language Model (LLM) based cold-start recommendation system.  
This repository contains the full Qwen-based ColdRAG pipeline implemented with **vLLM** for LLM inference and **BAAI/bge-large-en-v1.5** for embedding-based retrieval.

# Environment Setup

```bash
conda create -n coldrag python=3.10 -y
conda activate coldrag
pip install -r requirements.txt
```
# Starting the vLLM Server

ColdRAG only needs a single vLLM server, for the reasoning LLM. Embeddings
are loaded locally via `transformers` in the same process, not served over
HTTP.

## Qwen LLM Server
```bash
export VLLM_SERVER_URL=http://localhost:8000/v1/chat/completions
export VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct
export VLLM_MAX_MODEL_LEN=131072

python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --port 8000 \
  --gpu-memory-utilization 0.8 \
  --tensor-parallel-size 1 \
  --max-model-len 131072 \
  --rope-scaling '{"type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' \
  --download-dir "$CACHE_DIR"
```
`outlines==0.0.46` (vLLM's default guided-decoding backend) depends on a
`pyairports` package that is broken on PyPI (installs but has no importable
module), which makes every chat completion request fail with a 500 error.
If you hit `ModuleNotFoundError: No module named 'pyairports'` in the vLLM
server log, add `--guided-decoding-backend lm-format-enforcer` to the
command above to avoid that code path.

## Embedding Model
No server needed. `EMBED_MODEL` selects the local HuggingFace embedding model
and defaults to `BAAI/bge-large-en-v1.5`. **Leave this unset if you're using
the pre-built `rag_output_qwen/` index below** — the vector store was built
with the default model, and querying it with a different embedding model
silently breaks retrieval (entity/candidate matching returns 0 results with
no error). Only set `EMBED_MODEL` if you are also rebuilding the index from
scratch with that same model.
# Dataset

Place them like this after preprocessing:
```bash
ColdRAG/
├── dataset/
│     └── Video_Games/...
└── rag_output_qwen/
      └── Video_Games/...
```
## Notes
- If rag_output_qwen/Video_Games/ is empty, ColdRAG will first run knowledge graph construction (indexing) and then proceed to inference.
- If the RAG index exists, ColdRAG skips indexing and runs inference directly.

# Running ColdRAG
```bash
python main.py --model ColdRAG_qwen --dataset Video_Games --core 15 --cand_size 100 --k 10 --batch_size 5 \
  --out outputs/ColdRAG_Video_Games_core15.json
```
`--out` defaults to `./outputs/preds.json` if omitted.

# Output Files
After running ColdRAG, two output files are generated:
## Prediction Log File
```bash
outputs/ColdRAG_Video_Games_core15.json
```
## Evaluation File
```bash
outputs/ColdRAG_Video_Games_core15_eval.json
```

# Acknowledgements
ColdRAG's indexing and retrieval engine (the `coldrag/` package) is adapted
from [LightRAG](https://github.com/HKUDS/LightRAG) (HKUDS, MIT License; see
[`coldrag/LICENSE`](coldrag/LICENSE)). We build on it by replacing its
similarity-based retrieval with LLM-guided, query-adaptive multi-hop
traversal for item cold-start recommendation.
