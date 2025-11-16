# ColdRAG

ColdRAG is a Retrieval-Augmented, Large Language Model (LLM) based
cold-start recommender system using Qwen via vLLM.

## Setup

1. Install dependencies:
pip install -r requirements.txt
2. Download your dataset and place under:
ColdRAG/dataset/
3. Run vLLM server:
bash run.sh
4. Run ColdRAG:
python main.py --dataset Video_Games --core 15 --k 10
