# filename: vllm_preset.py
from __future__ import annotations
import os
import threading
from typing import Any, List, Optional
import asyncio
import aiohttp

# -----------------------------
# Config (exported constants)
# -----------------------------
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-large-en-v1.5")
# VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen2.5-32B-Instruct")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
VLLM_SERVER_URL = os.environ.get("VLLM_SERVER_URL", "http://localhost:8000/v1/chat/completions")

# _DEF_MAX_TOKENS = int(os.environ.get("VLLM_MAX_TOKENS", "256"))
_DEF_MAX_TOKENS = int(os.environ.get("VLLM_MAX_TOKENS",
                                     os.environ.get("VLLM_MAX_MODEL_LEN", "131072"))) // 2

_DEF_TEMP = float(os.environ.get("VLLM_TEMPERATURE", "0.0"))
_DEF_TOP_P = float(os.environ.get("VLLM_TOP_P", "1.0"))
_DEF_TOP_K = int(os.environ.get("VLLM_TOP_K", "1"))
_DEF_REP_PEN = float(os.environ.get("VLLM_REP_PENALTY", "1.1"))

# -----------------------------
# vLLM (REMOTE CLIENT MODE)
# -----------------------------
async def _vllm_complete_async(prompt: str, **kwargs: Any) -> str:
    """Async call to the running vLLM server (OpenAI-compatible API)."""
    max_tokens = int(kwargs.get("max_tokens", _DEF_MAX_TOKENS))
    temperature = float(kwargs.get("temperature", _DEF_TEMP))
    top_p = float(kwargs.get("top_p", _DEF_TOP_P))
    top_k = int(kwargs.get("top_k", _DEF_TOP_K))
    repetition_penalty = float(kwargs.get("repetition_penalty", _DEF_REP_PEN))
    stop = kwargs.get("stop") or kwargs.get("stop_sequences") or None

    payload = {
        "model": VLLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "n": 1,
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(VLLM_SERVER_URL, json=payload) as resp:
            data = await resp.json()
            text = data["choices"][0]["message"]["content"].strip()
            if stop:
                for s in stop:
                    if s and s in text:
                        text = text.split(s)[0]
            return text

def vllm_qwen_complete(prompt: str, **kwargs: Any) -> str:
    """
    Synchronous wrapper to call the async vLLM completion endpoint.
    """
    return asyncio.run(_vllm_complete_async(prompt, **kwargs))

# -----------------------------
# Embedding (BGE via HF)
# -----------------------------
__EMB_LOCK = threading.Lock()
__EMB_TOKENIZER = None
__EMB_MODEL = None
__EMB_DIM = None

def _init_embedder():
    """Lazy-load a single embedding model and tokenizer; set global dim."""
    global __EMB_TOKENIZER, __EMB_MODEL, __EMB_DIM
    if __EMB_MODEL is not None:
        return
    import torch
    from transformers import AutoModel, AutoTokenizer
    __EMB_TOKENIZER = AutoTokenizer.from_pretrained(EMBED_MODEL, trust_remote_code=True)
    __EMB_MODEL = AutoModel.from_pretrained(
        EMBED_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    ).eval()
    cfg = getattr(__EMB_MODEL, "config", None)
    __EMB_DIM = getattr(cfg, "hidden_size", None)
    if __EMB_DIM is None:
        with torch.no_grad():
            tmp = __EMB_TOKENIZER(["dim probe"], return_tensors="pt").to(__EMB_MODEL.device)
            out = __EMB_MODEL(**tmp).last_hidden_state
            __EMB_DIM = int(out.shape[-1])

def _mean_pool(last_hidden_state, attention_mask):
    import torch
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-6)
    return summed / counts

def vllm_bge_embed(texts: List[str] | str, normalize: bool = True) -> List[List[float]]:
    """
    Functional embedding helper. Returns a list of vectors (L2-normalized by default).
    """
    import torch
    if isinstance(texts, str):
        texts = [texts]
    with __EMB_LOCK:
        _init_embedder()
    tok = __EMB_TOKENIZER
    model = __EMB_MODEL
    device = next(model.parameters()).device

    with torch.no_grad():
        batch = tok(texts, return_tensors="pt", padding=True, truncation=True).to(device)
        out = model(**batch)
        emb = _mean_pool(out.last_hidden_state, batch["attention_mask"])
        if normalize:
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb.float().cpu().tolist()

class VLLMEmbedWrapper:
    """
    Wrapper around vllm_bge_embed that adds an .embedding_dim attribute
    and async compatibility for LightRAG/NanoVectorDB.
    """

    def __init__(self, embedding_dim: int = 1024, normalize: bool = True):
        self.embedding_func = vllm_bge_embed
        self.embedding_dim = embedding_dim
        self.normalize = normalize

    @property
    def func(self):
        # Expose a .func handle for backward compatibility
        return self.__call__

    async def __call__(self, texts: List[str] | str) -> List[List[float]]:
        """Asynchronous call wrapper for LightRAG."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embedding_func, texts)

# ---- Embedder object for LightRAG/NanoVectorDB (has .embedding_dim) ----
class _BGEEmbedder:
    def __init__(self, model_name: Optional[str] = None, normalize: bool = True):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name or EMBED_MODEL
        self.normalize = normalize

        self.tok = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        ).eval()
        cfg = getattr(self.model, "config", None)
        dim = getattr(cfg, "hidden_size", None)
        if dim is None:
            with torch.no_grad():
                tmp = self.tok(["dim probe"], return_tensors="pt").to(self.model.device)
                dim = int(self.model(**tmp).last_hidden_state.shape[-1])
        self.embedding_dim = int(dim)

    def __call__(self, texts: List[str] | str) -> List[List[float]]:
        import torch
        if isinstance(texts, str):
            texts = [texts]
        batch = self.tok(texts, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        with torch.no_grad():
            out = self.model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(out.dtype)
            vec = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
            if self.normalize:
                vec = torch.nn.functional.normalize(vec, p=2, dim=1)
        return vec.float().cpu().tolist()

def make_bge_embedder(model_name: Optional[str] = None, normalize: bool = True):
    """Factory for an embedder callable with `.embedding_dim` attribute (LightRAG compatible)."""
    return _BGEEmbedder(model_name=model_name, normalize=normalize)

__all__ = [
    "EMBED_MODEL",
    "VLLM_MODEL",
    "vllm_qwen_complete",
    "vllm_bge_embed",
    "make_bge_embedder",
]
