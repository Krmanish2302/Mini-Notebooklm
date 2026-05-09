"""
base_embedder.py

Abstract base class for all embedders.
Every embedder must implement `embed` (batch) and `embed_single` (single text).
"""
from abc import ABC, abstractmethod
from typing import List
import numpy as np


class BaseEmbedder(ABC):
    """Contract that all embedder implementations must satisfy."""

    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed a batch of texts.

        Args:
            texts: List of strings to embed.

        Returns:
            numpy array of shape (len(texts), embedding_dim).
        """
        ...

    @abstractmethod
    def embed_single(self, text: str) -> np.ndarray:
        """
        Embed a single string.

        Args:
            text: The string to embed.

        Returns:
            numpy array of shape (embedding_dim,).
        """
        ...

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality of the output embeddings."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier."""
        ...
