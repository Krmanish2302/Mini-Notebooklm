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
        # Default fallback to env or all-MiniLM-L6-v2
        if not model_name:
            model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
            
        # Standardize names/aliases
        if model_name == "huggingface":
            model_name = "all-MiniLM-L6-v2"
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