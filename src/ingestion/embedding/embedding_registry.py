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
    def get(cls, provider: str = None):
        provider = provider or os.getenv("EMBEDDING_PROVIDER", "huggingface")
        if provider in cls._instances:
            return cls._instances[provider]

        if provider == "huggingface":
            from langchain_community.embeddings import HuggingFaceEmbeddings
            emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        elif provider == "openai":
            from langchain_openai import OpenAIEmbeddings
            emb = OpenAIEmbeddings(model="text-embedding-3-small")
        else:
            raise ValueError(f"Unknown embedding provider '{provider}'. Use 'openai' or 'huggingface'.")

        cls._instances[provider] = emb
        return emb