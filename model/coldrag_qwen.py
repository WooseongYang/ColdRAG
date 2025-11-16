# model/coldrag_qwen.py
import os, json, re, asyncio
from typing import Any, List
import pandas as pd
import numpy as np
from tqdm import tqdm

# from lightrag import LightRAG, QueryParam
# from lightrag.kg.shared_storage import initialize_pipeline_status

from coldrag import LightRAG, QueryParam
from coldrag.kg.shared_storage import initialize_pipeline_status

# keep your current vllm_preset exactly as-is
from vllm_preset import vllm_qwen_complete, VLLMEmbedWrapper
# ^ if your current vllm_preset doesn't export make_bge_embedder, switch to the wrapper class it provides
#   as long as it has .embedding_dim and is callable.

# ---------- helper: turn sync callables into awaitables ----------
def _ensure_async(fn):
    """Return an awaitable version of a callable (no-op if already async)."""
    if asyncio.iscoroutinefunction(fn):
        return fn
    async def _aw(*args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))
    return _aw

def _extract_topk_lines(text: str, k: int) -> List[str]:
    lines = (text or "").strip().splitlines()
    out: List[str] = []
    for ln in lines:
        m = re.match(r"^\s*\d+\.\s*(.+)$", ln.strip())
        if m:
            out.append(m.group(1).strip())
            if len(out) >= k:
                break
    return out

class ColdRAG_qwen:
    def __init__(self, dataset, core, candidate_list_size, mode):
        self.dataset = dataset
        self.core = core
        self.candidate_list_size = candidate_list_size
        self.mode = mode

        self.processed_dir = f"./dataset/{dataset}/processed"
        self.data_path = f"{self.processed_dir}/data_eval_{core}.json"
        self.metadata_path = f"{self.processed_dir}/metadata_{core}core.csv"

        # # Build async-compatible callables for LightRAG
        # embedder_obj = VLLMEmbedWrapper(embedding_dim=1024)
        # # callable with .embedding_dim (sync)
        # self.embedding_func = _ensure_async(embedder_obj)

        self.embedding_func = VLLMEmbedWrapper(embedding_dim=1024)

        # vllm_qwen_complete may already be async in your current vllm_preset; this is safe either way
        self.llm_func = _ensure_async(vllm_qwen_complete)

        self.rag: LightRAG | None = None
        self.candidate_lists: dict[str, list[str]] = {}
        self.sampled_sequences: list[dict[str, Any]] = []
        self.metadata = pd.DataFrame()

    async def initialize(self):
        if self.rag is not None:
            return

        # Construct LightRAG with async-able funcs
        self.rag = LightRAG(
            working_dir=f"./rag_output_qwen/{self.dataset}",
            embedding_func=self.embedding_func,
            llm_model_func=self.llm_func,
            enable_llm_cache=False,
        )

        # REQUIRED for async pipeline (indexing + KG building)
        await self.rag.initialize_storages()
        await initialize_pipeline_status()

        # load data
        with open(self.data_path, "r", encoding="utf-8") as f:
            self.sampled_sequences = json.load(f)
        self.metadata = pd.read_csv(self.metadata_path)

        cand_path = f"{self.processed_dir}/candidate_list_{self.candidate_list_size}_{self.core}.json"
        if os.path.exists(cand_path) and os.path.getsize(cand_path) > 0:
            with open(cand_path, "r", encoding="utf-8") as f:
                self.candidate_lists = json.load(f)
        else:
            self.candidate_lists = {}

    async def run_indexing(self, limit: int | None = None):
        """Use LightRAG's async insert; this is where the embedder MUST be awaitable."""
        assert self.rag is not None
        input_dir = f"{self.processed_dir}/item_text_{self.core}"
        if not os.path.isdir(input_dir):
            print(f"[ColdRAG_qwen] Indexing skipped (missing dir): {input_dir}")
            return

        files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".txt")]
        if limit:
            files = files[:limit]
        # files = ['./dataset/Video_Games/processed/item_text_15/batch_0.txt', './dataset/Video_Games/processed/item_text_15/batch_0.txt']
        print(f"[ColdRAG_qwen] Inserting {len(files)} item docs into LightRAG...")
        for fp in tqdm(files, desc="Indexing"):
            try:
                with open(fp, "r", encoding="utf-8") as fh:
                    await self.rag.ainsert(
                        fh.read(),
                        split_by_character="\n\n###",
                        split_by_character_only=True,
                        file_paths=fp,
                    )
            except Exception as e:
                print(f"[ColdRAG_qwen] insert error {os.path.basename(fp)}: {e}")

    async def _aquery_once(self, prompt: str, k: int, max_retry: int = 5) -> List[str]:
        assert self.rag is not None
        for t in range(max_retry):
            try:
                resp = await self.rag.aquery(prompt, param=QueryParam(mode=self.mode, enable_rerank=False))
                recs = _extract_topk_lines(resp, k)
                if len(recs) >= k:
                    return recs
                print(f"[ColdRAG_qwen] retry {t+1}: got {len(recs)}")
            except Exception as e:
                print(f"[ColdRAG_qwen] query error (try {t+1}): {e}")
            await asyncio.sleep(1.2)
        return ["UNKNOWN"] * k

    async def run_coldrag(self, output_path: str, k: int, batch_size: int = 20):
        assert self.rag is not None
        prompts: List[str] = []
        for entry in self.sampled_sequences:
            hist = entry.get("input", [])[-20:]
            uid = str(entry.get("user_id", ""))
            cand = self.candidate_lists.get(uid, [])
            cand_text = "\n".join(f"{i+1}. {t}" for i, t in enumerate(cand))
            if self.mode == "coldrag":
                prompt = (
                    f"I've purchased the following products in the past in order:\n{hist}\n\n"
                    f"Please carefully recommend top-{k} products among the candidate products by how likely I am to purchase them next, "
                    f"based on my past purchasing history.\n"
                    f"Think step by step, but only output the final ranking in the following format:\n\n"
                    f"1. <product name>\n2. <product name>\n...\n\n"
                    f"Only include items from the given candidate list. Do not add explanations or any other text."
                )
            else:
                prompt = (
                    f"I've purchased the following products in the past in order:\n{hist}\n\n"
                    f"Now there are {len(cand)} candidate products that I can consider purchasing next:\n{cand_text}\n\n"
                    f"Please carefully recommend top-{k} products among the candidate products by how likely I am to purchase them next, "
                    f"based on my past purchasing history.\n"
                    f"Think step by step, but only output the final ranking in the following format:\n\n"
                    f"1. <product name>\n2. <product name>\n...\n\n"
                    f"Only include items from the given candidate list. Do not add explanations or any other text."
                )
            prompts.append(prompt)

        print(f"[ColdRAG_qwen] Querying {len(prompts)} users (mode={self.mode})...")
        results_all: List[List[str]] = []
        for i in tqdm(range(0, len(prompts), batch_size), desc="Query Batches"):
            batch = prompts[i : i + batch_size]
            tasks = [self._aquery_once(p, k) for p in batch]
            results = await asyncio.gather(*tasks)
            results_all.extend(results)

        out = []
        for entry, preds in zip(self.sampled_sequences, results_all):
            e = dict(entry)
            e["predicted_items"] = preds
            out.append(e)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        return out

    @staticmethod
    def evaluate_coldrag(data: list[dict[str, Any]], k: int):
        recall_cnt = 0
        ndcg_sum = 0.0
        mrr_sum = 0.0
        for e in data:
            true_item = e.get("true_title", "")
            preds = (e.get("predicted_items", []) or [])[:k]
            if true_item in preds:
                r = preds.index(true_item)
                recall_cnt += 1
                ndcg_sum += 1 / np.log2(r + 2)
                mrr_sum += 1 / (r + 1)
        n = len(data) if data else 0
        return (
            (recall_cnt / n) if n else 0.0,
            float(ndcg_sum / n) if n else 0.0,
            (mrr_sum / n) if n else 0.0,
        )
