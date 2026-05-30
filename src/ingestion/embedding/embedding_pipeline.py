"""
embedding_pipeline.py

MD5-cached embedding pipeline wrapping TextEmbedder.
Used by retrieval layer (HybridRetriever) for query-time embedding.
Ingestion-time embedding goes through LangChain FAISS directly.
"""
from __future__ import annotations
import hashlib
import logging
from typing import Dict, List
import numpy as np
from .text_embedder import TextEmbedder

logger = logging.getLogger(__name__)


class EmbeddingPipeline:
    def __init__(self, use_cache: bool = True):
        self.embedder  = TextEmbedder()
        self.use_cache = use_cache
        self._cache: Dict[str, np.ndarray] = {}

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query — canonical method for retrieval."""
        key = hashlib.md5(query.encode()).hexdigest()
        if self.use_cache and key in self._cache:
            return self._cache[key]
        vec = self.embedder.embed_single(query)
        if self.use_cache:
            self._cache[key] = vec
        return vec

    # alias for backward-compat
    embed_single = embed_query

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of texts. Returns shape (N, dim)."""
        dim = self.embedder.dimension
        if not texts:
            return np.empty((0, dim), dtype="float32")

        if not self.use_cache:
            return np.atleast_2d(self.embedder.embed(texts))

        cached, to_embed, indices = [], [], []
        for i, t in enumerate(texts):
            k = hashlib.md5(t.encode()).hexdigest()
            if k in self._cache:
                cached.append((i, self._cache[k]))
            else:
                to_embed.append(t)
                indices.append(i)

        if to_embed:
            new_vecs = np.atleast_2d(self.embedder.embed(to_embed))
            for idx, text, vec in zip(indices, to_embed, new_vecs):
                k = hashlib.md5(text.encode()).hexdigest()
                self._cache[k] = vec
                cached.append((idx, vec))

        cached.sort(key=lambda x: x[0])
        return np.stack([r[1] for r in cached])

    def clear_cache(self) -> None:
        self._cache.clear()