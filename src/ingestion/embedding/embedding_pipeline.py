"""
embedding_pipeline.py  —  EmbeddingPipeline

Fixes vs original:
    - embed_batch() always returns shape (N, dim), even for empty input
      (was returning (0,) for empty lists causing downstream shape errors).
    - expose embed_query() as the canonical single-query entry point;
      embed_single aliased to same so both callers work.
    - MD5 cache uses full text as key to avoid collision on short texts.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Dict, List, Any

import numpy as np

from .text_embedder import TextEmbedder

logger = logging.getLogger(__name__)


class EmbeddingPipeline:
    """Manages embedding with semantic caching."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_cache: bool = True):
        self.model_name = model_name
        self.embedder = TextEmbedder(model_name)
        self.use_cache = use_cache
        self._cache: Dict[str, np.ndarray] = {}

    # ── public API ────────────────────────────────────────────────────────────

    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Embed a list of chunk dicts; attaches 'embedding' to each chunk."""
        texts = [c["content"] for c in chunks]
        embeddings = self.embed_batch(texts)  # shape (N, dim)
        for chunk, emb in zip(chunks, embeddings):
            chunk["embedding"] = emb.tolist()
        return chunks

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Embed a batch of texts.  Always returns shape (N, dim) — never (dim,)
        or (0,) for empty input.

        Fix (Bug 7): empty list now returns (0, dim) not (0,).
        """
        dim = self.embedder.dimension

        if not texts:
            # Fix Bug 7 — return correct 2-D empty array
            return np.empty((0, dim), dtype="float32")

        if not self.use_cache:
            result = self.embedder.embed(texts)
            return np.atleast_2d(result)

        cached_results: List[tuple] = []
        to_embed_texts: List[str] = []
        to_embed_indices: List[int] = []

        for i, text in enumerate(texts):
            key = hashlib.md5(text.encode()).hexdigest()
            if key in self._cache:
                cached_results.append((i, self._cache[key]))
            else:
                to_embed_texts.append(text)
                to_embed_indices.append(i)

        if to_embed_texts:
            new_embs = np.atleast_2d(self.embedder.embed(to_embed_texts))
            for idx, text, emb in zip(to_embed_indices, to_embed_texts, new_embs):
                key = hashlib.md5(text.encode()).hexdigest()
                self._cache[key] = emb
                cached_results.append((idx, emb))

        cached_results.sort(key=lambda x: x[0])
        return np.stack([r[1] for r in cached_results])  # always (N, dim)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string.  Returns 1-D array of shape (dim,).
        This is the canonical method called by HybridRetriever.
        """
        return self.embedder.embed_single(query)

    # alias kept for backwards compatibility
    embed_single = embed_query

    def clear_cache(self) -> None:
        self._cache.clear()
