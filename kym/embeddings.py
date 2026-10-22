"""Sentence embeddings (fastembed, ONNX) with a persistent cache.

KYM_EMBEDDINGS=hash switches to a deterministic hashing embedder: no model download, used by the tests.
"""

from __future__ import annotations

import hashlib
import os
import re
import threading

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import get_settings
from .models import EmbeddingCache

DIM = 384
_model = None
_model_lock = threading.Lock()
_memory: dict[str, np.ndarray] = {}


def _backend() -> str:
    return os.getenv("KYM_EMBEDDINGS", "fastembed")


def _key(text: str) -> str:
    name = get_settings().embed_model if _backend() == "fastembed" else "hash"
    return hashlib.sha256(f"{name}|{text}".encode()).hexdigest()


def _hash_embed(texts: list[str]) -> np.ndarray:
    out = np.zeros((len(texts), DIM), dtype=np.float32)
    for i, text in enumerate(texts):
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        grams = tokens + [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]
        for gram in grams:
            h = int.from_bytes(hashlib.md5(gram.encode()).digest()[:8], "little")
            out[i, h % DIM] += 1.0 if (h >> 40) & 1 else -1.0
    return out


def _model_embed(texts: list[str]) -> np.ndarray:
    global _model
    with _model_lock:
        if _model is None:
            from fastembed import TextEmbedding

            _model = TextEmbedding(get_settings().embed_model)
        return np.array(list(_model.embed(texts)), dtype=np.float32)


def _normalise(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return v / norms


def embed(texts: list[str], session: Session | None = None) -> np.ndarray:
    """Unit-length embeddings, one row per text. Pass a session to persist the cache across runs."""
    if not texts:
        return np.zeros((0, DIM), dtype=np.float32)
    keys = [_key(t) for t in texts]
    found: dict[str, np.ndarray] = {k: _memory[k] for k in keys if k in _memory}
    missing = [k for k in dict.fromkeys(keys) if k not in found]
    if missing and session is not None:
        for start in range(0, len(missing), 500):
            chunk = missing[start:start + 500]
            for row in session.scalars(select(EmbeddingCache).where(EmbeddingCache.key.in_(chunk))):
                found[row.key] = np.frombuffer(row.vector, dtype=np.float32)
                _memory[row.key] = found[row.key]
    todo = {k: t for k, t in zip(keys, texts) if k not in found}
    if todo:
        vectors = _hash_embed(list(todo.values())) if _backend() == "hash" else _model_embed(list(todo.values()))
        vectors = _normalise(vectors)
        for k, v in zip(todo, vectors):
            found[k] = v
            _memory[k] = v
            if session is not None:
                session.add(EmbeddingCache(key=k, vector=v.astype(np.float32).tobytes()))
    return np.vstack([found[k] for k in keys])


def calibrate(sim: np.ndarray | float):
    """Map raw cosine onto 0..1. bge-small scores unrelated phrases around 0.45, related ones 0.65-0.8."""
    lo, hi = (0.05, 0.6) if _backend() == "hash" else (0.55, 0.80)
    return np.clip((np.asarray(sim) - lo) / (hi - lo), 0.0, 1.0)
