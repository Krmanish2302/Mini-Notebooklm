"""
embedding_registry.py

Registry mapping provider names → LangChain Embeddings instances.
"""
from __future__ import annotations
import os
from typing import Dict


class EmbeddingRegistry:
    _instances: Dict[str, object] = {}

    @classmethod
    def get(cls, model_name: str = None):
        # Default fallback to env or sentence-transformers/multi-qa-mpnet-base-dot-v1
        if not model_name:
            model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/multi-qa-mpnet-base-dot-v1")
            
        # Standardize names/aliases
        if model_name in ("huggingface", "multi-qa-mpnet-base"):
            model_name = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
        elif model_name == "openai":
            model_name = "text-embedding-3-small"

        if model_name in cls._instances:
            return cls._instances[model_name]

        if model_name in {"text-embedding-3-small", "text-embedding-3-large"}:
            from langchain_openai import OpenAIEmbeddings
            emb = OpenAIEmbeddings(model=model_name)
        else:
            # Assume it's a HuggingFace model name
            from langchain_community.embeddings import HuggingFaceEmbeddings
            emb = HuggingFaceEmbeddings(model_name=model_name)

        cls._instances[model_name] = emb
        return emb

    @classmethod
    def get_default(cls):
        return cls.get()

    @classmethod
    def get_by_dim(cls, dim: int):
        dim_map = {
            384: "all-MiniLM-L6-v2",
            768: "sentence-transformers/multi-qa-mpnet-base-dot-v1",
            1024: "e5-large-v2",
            1536: "text-embedding-3-small",
            3072: "text-embedding-3-large"
        }
        model_name = dim_map.get(dim)
        if model_name:
            return cls.get(model_name)
        return None