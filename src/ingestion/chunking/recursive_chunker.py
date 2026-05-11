"""
recursive_chunker.py — Recursive character-based text splitter.
Compatible with LangChain v0.3+ (langchain_text_splitters package).
"""
from __future__ import annotations

from typing import List

# LangChain v0.3+: text splitters moved to langchain_text_splitters
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    # Fallback for older installs
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore

from .base_chunker import BaseChunker


class RecursiveChunker(BaseChunker):
    """
    Splits text recursively on paragraph, sentence, word, and character
    boundaries until every chunk is within chunk_size tokens.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separators: List[str] | None = None,
    ) -> None:
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators    = separators or ["\n\n", "\n", ". ", " ", ""]
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
        )

    def chunk(self, text: str, source_id: str = "") -> List[dict]:
        if not text or not text.strip():
            return []
        raw_chunks = self._splitter.split_text(text)
        return [
            {
                "chunk_id":  f"{source_id}_rc_{i}",
                "source_id": source_id,
                "text":      chunk,
                "index":     i,
            }
            for i, chunk in enumerate(raw_chunks)
        ]
