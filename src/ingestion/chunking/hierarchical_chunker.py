"""
hierarchical_chunker.py — Multi-level parent/child chunk decomposition.
Compatible with LangChain v0.3+ (langchain_text_splitters package).
"""
from __future__ import annotations

from typing import List

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore

from .base_chunker import BaseChunker


class HierarchicalChunker(BaseChunker):
    """
    Produces two levels of chunks:
      - Parent chunks (large context window, e.g. 1024 tokens)
      - Child chunks  (fine-grained retrieval units, e.g. 256 tokens)

    Each child chunk carries a parent_chunk_id reference so the retriever
    can optionally expand retrieved child chunks to their parent context.
    """

    def __init__(
        self,
        parent_chunk_size: int = 1024,
        child_chunk_size:  int = 256,
        chunk_overlap:     int = 32,
        max_depth:         int = 3,
    ) -> None:
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size  = child_chunk_size
        self.chunk_overlap     = chunk_overlap
        self.max_depth         = max_depth

        self._parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.parent_chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        self._child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.child_chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    def chunk(self, text: str, source_id: str = "") -> List[dict]:
        if not text or not text.strip():
            return []

        chunks: List[dict] = []
        parent_texts = self._parent_splitter.split_text(text)

        for p_idx, parent_text in enumerate(parent_texts):
            parent_id = f"{source_id}_par_{p_idx}"
            # Include parent chunk itself
            chunks.append({
                "chunk_id":        parent_id,
                "source_id":       source_id,
                "text":            parent_text,
                "index":           p_idx,
                "level":           "parent",
                "parent_chunk_id": None,
            })
            # Split into child chunks
            child_texts = self._child_splitter.split_text(parent_text)
            for c_idx, child_text in enumerate(child_texts):
                chunks.append({
                    "chunk_id":        f"{parent_id}_ch_{c_idx}",
                    "source_id":       source_id,
                    "text":            child_text,
                    "index":           c_idx,
                    "level":           "child",
                    "parent_chunk_id": parent_id,
                })

        return chunks
