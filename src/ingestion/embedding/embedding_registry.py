"""
embedding_registry.py  —  Singleton registry for embedding models

Additions vs original:
    - get_by_dim(dim): reverse lookup — given a FAISS index dimension,
      return the EmbeddingPipeline that produced it.  Required for
      ChatGraph to know which model to use when querying each index.
    - register(model_name, dim): explicit dim registration.
    - Thread-safe singleton per model_name.
"""
from __future__ import annotations

import logging
import threading
from typing import Dict, List, Optional

from .embedding_pipeline import EmbeddingPipeline

logger = logging.getLogger(__name__)

# Known model → dimension mapping (extend as needed)
_MODEL_DIMS: Dict[str, int] = {
    "all-MiniLM-L6-v2":      384,
    "all-MiniLM-L12-v2":     384,
    "all-mpnet-base-v2":     768,
    "multi-qa-mpnet-base-v2": 768,
    "e5-large-v2":           1024,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}


class EmbeddingRegistry:
    """
    Thread-safe singleton cache of EmbeddingPipeline instances.

    Usage:
        pipeline = EmbeddingRegistry.get("all-MiniLM-L6-v2")
        pipeline = EmbeddingRegistry.get_by_dim(768)   # → mpnet pipeline
    """

    _lock = threading.Lock()
    _by_name: Dict[str, EmbeddingPipeline] = {}
    _by_dim:  Dict[int, EmbeddingPipeline] = {}  # dim → pipeline

    @classmethod
    def get(cls, model_name: str) -> EmbeddingPipeline:
        """
        Return (or create) an EmbeddingPipeline for *model_name*.
        Subsequent calls with the same name return the cached instance.
        """
        with cls._lock:
            if model_name not in cls._by_name:
                logger.info("EmbeddingRegistry: loading model %s", model_name)
                pipeline = EmbeddingPipeline(model_name=model_name)
                cls._by_name[model_name] = pipeline
                # Register dim reverse-lookup
                dim = _MODEL_DIMS.get(model_name)
                if dim and dim not in cls._by_dim:
                    cls._by_dim[dim] = pipeline
                    logger.info("EmbeddingRegistry: dim %d → %s", dim, model_name)
            return cls._by_name[model_name]

    @classmethod
    def get_by_dim(cls, dim: int) -> Optional[EmbeddingPipeline]:
        """
        Return the EmbeddingPipeline whose output dimension is *dim*.
        Returns None if no model for that dim has been loaded yet.
        """
        with cls._lock:
            return cls._by_dim.get(dim)

    @classmethod
    def register_dim(cls, model_name: str, dim: int) -> None:
        """Explicitly register a (model_name, dim) pair for custom models."""
        with cls._lock:
            if model_name in cls._by_name and dim not in cls._by_dim:
                cls._by_dim[dim] = cls._by_name[model_name]

    @classmethod
    def list_models(cls) -> List[str]:
        return list(cls._by_name.keys())

    @classmethod
    def available_models(cls) -> Dict[str, int]:
        """Return {model_name: dim} for all known models (loaded or not)."""
        return dict(_MODEL_DIMS)

    @classmethod
    def clear(cls) -> None:
        """Clear all cached instances (useful for testing)."""
        with cls._lock:
            cls._by_name.clear()
            cls._by_dim.clear()
