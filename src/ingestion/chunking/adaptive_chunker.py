"""
adaptive_chunker.py — Selects chunking strategy based on document type.
Compatible with LangChain v0.3+ (langchain_text_splitters package).
"""
from __future__ import annotations

from typing import List

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore

from .base_chunker import BaseChunker
from .paragraph_chunker import ParagraphChunker
from .page_chunker import PageChunker


class AdaptiveChunker(BaseChunker):
    """
    Routes to the most appropriate chunker based on source_type hint.

    source_type mappings:
        pdf       -> PageChunker
        website   -> ParagraphChunker
        youtube   -> ParagraphChunker
        text / md -> RecursiveCharacterTextSplitter (default)
        *         -> RecursiveCharacterTextSplitter (fallback)
    """

    _STRATEGY_MAP: dict = {
        "pdf":     "page",
        "website": "paragraph",
        "youtube": "paragraph",
    }

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap
        self._default_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self._page_chunker      = PageChunker()
        self._paragraph_chunker = ParagraphChunker()

    def chunk(
        self,
        text: str,
        source_id: str = "",
        source_type: str = "text",
    ) -> List[dict]:
        if not text or not text.strip():
            return []

        strategy = self._STRATEGY_MAP.get(source_type.lower(), "recursive")

        if strategy == "page":
            return self._page_chunker.chunk(text, source_id=source_id)
        if strategy == "paragraph":
            return self._paragraph_chunker.chunk(text, source_id=source_id)

        # Default: recursive
        raw_chunks = self._default_splitter.split_text(text)
        return [
            {
                "chunk_id":  f"{source_id}_adp_{i}",
                "source_id": source_id,
                "text":      chunk,
                "index":     i,
            }
            for i, chunk in enumerate(raw_chunks)
        ]
