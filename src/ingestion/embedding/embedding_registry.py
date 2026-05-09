"""
embedding_registry.py

Registry of available embedding models.
Allows swapping models at runtime without changing calling code.

Usage:
    embedder = EmbeddingRegistry.get("all-MiniLM-L6-v2")
    vecs = embedder.embed(["hello world"])
"""
from typing import Dict, Type
from .base_embedder import BaseEmbedder
from .text_embedder import TextEmbedder


class EmbeddingRegistry:
    """
    Central registry mapping model-name aliases to TextEmbedder instances.

    All listed names resolve to HuggingFace sentence-transformer models.
    Add entries here to expose new models to the rest of the pipeline.
    """

    # Canonical name → model path on HuggingFace Hub
    _MODEL_MAP: Dict[str, str] = {
        # Fast, lightweight — default for most use-cases
        "all-MiniLM-L6-v2":   "sentence-transformers/all-MiniLM-L6-v2",
        "minilm":              "sentence-transformers/all-MiniLM-L6-v2",
        # Balanced quality / speed
        "all-mpnet-base-v2":  "sentence-transformers/all-mpnet-base-v2",
        "mpnet":               "sentence-transformers/all-mpnet-base-v2",
        # Highest quality, multilingual
        "e5-large-v2":         "intfloat/e5-large-v2",
        "e5":                  "intfloat/e5-large-v2",
        # Multilingual
        "multilingual-e5-large": "intfloat/multilingual-e5-large",
    }

    # Singleton cache — one instance per model name
    _instances: Dict[str, BaseEmbedder] = {}

    @classmethod
    def get(cls, model_name: str = "all-MiniLM-L6-v2") -> BaseEmbedder:
        """
        Return a (cached) embedder for *model_name*.

        Args:
            model_name: Alias or full HuggingFace path.

        Returns:
            BaseEmbedder instance.

        Raises:
            ValueError: If the alias is not registered.
        """
        key = model_name.strip().lower()
        if key in cls._instances:
            return cls._instances[key]

        # Resolve alias → full model path
        resolved = cls._MODEL_MAP.get(key, model_name)  # fall through for full paths

        embedder = TextEmbedder(resolved)
        cls._instances[key] = embedder
        return embedder

    @classmethod
    def list_models(cls) -> list:
        """Return all registered model aliases."""
        return sorted(cls._MODEL_MAP.keys())

    @classmethod
    def clear_cache(cls) -> None:
        """Free all cached model instances (useful for testing)."""
        cls._instances.clear()
