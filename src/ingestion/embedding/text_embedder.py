"""
text_embedder.py

LangChain-backed embedder. Supports OpenAI and HuggingFace.
EMBEDDING_PROVIDER env var selects the provider.
"""
from __future__ import annotations
import os
import numpy as np
from typing import List
from .base_embedder import BaseEmbedder

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai")


class TextEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = None):
        self.model_name = model_name
        self._lc_embedder = self._build()

    def _build(self):
        if EMBEDDING_PROVIDER == "huggingface":
            from langchain_community.embeddings import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(
                model_name=self.model_name or "all-MiniLM-L6-v2"
            )
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=self.model_name or "text-embedding-3-small")

    def embed(self, texts: List[str]) -> np.ndarray:
        vecs = self._lc_embedder.embed_documents(texts)
        return np.array(vecs, dtype="float32")

    def embed_single(self, text: str) -> np.ndarray:
        vec = self._lc_embedder.embed_query(text)
        return np.array(vec, dtype="float32")

    @property
    def dimension(self) -> int:
        return len(self.embed_single("ping"))