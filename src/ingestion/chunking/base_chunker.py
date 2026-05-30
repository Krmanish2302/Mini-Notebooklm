"""base_chunker.py — Abstract base class for all chunkers."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document


class BaseChunker(ABC):
    @abstractmethod
    def chunk_documents(self, docs: List[Document]) -> List[Document]:
        """Split a list of Documents into smaller chunks."""