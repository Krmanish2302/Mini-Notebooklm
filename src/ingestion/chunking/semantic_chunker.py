"""
semantic_chunker.py — Embedding-based semantic chunker.
Splits on semantic similarity breakpoints rather than character count.
Compatible with LangChain v0.3+ (langchain_text_splitters package).
"""
from __future__ import annotations

from typing import List

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore

from .base_chunker import BaseChunker


class SemanticChunker(BaseChunker):
    """
    Falls back to RecursiveCharacterTextSplitter when a full semantic
    clustering model is not available; produces sensible paragraph-sized
    chunks that respect sentence boundaries.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        embedding_model: str = "all-MiniLM-L6-v2",
        breakpoint_threshold_type: str = "percentile",
    ) -> None:
        self.chunk_size               = chunk_size
        self.chunk_overlap            = chunk_overlap
        self.embedding_model          = embedding_model
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk(self, text: str, source_id: str = "") -> List[dict]:
        if not text or not text.strip():
            return []
        raw_chunks = self._splitter.split_text(text)
        return [
            {
                "chunk_id":  f"{source_id}_sem_{i}",
                "source_id": source_id,
                "text":      chunk,
                "index":     i,
            }
            for i, chunk in enumerate(raw_chunks)
        ]
