"""base_embedder.py — Abstract base for all embedders."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
import numpy as np


class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts. Returns shape (N, dim)."""

    @abstractmethod
    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text. Returns shape (dim,)."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding vector dimension."""