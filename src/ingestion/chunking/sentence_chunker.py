"""
sentence_chunker.py — Sentence-level chunker.
Splits text into individual sentences, then groups them into
fixed-size windows to respect context boundaries.
Compatible with LangChain v0.3+ (no LangChain dependency needed here).
"""
from __future__ import annotations

import re
from typing import List

from .base_chunker import BaseChunker

# Simple sentence boundary regex (handles Mr./Dr./etc. imperfectly but
# avoids requiring nltk at runtime)
_SENT_RE = re.compile(r'(?<=[.!?])\s+')


class SentenceChunker(BaseChunker):
    """
    Groups sentences into windows of `sentences_per_chunk` with
    `overlap` sentences of context carry-over.
    """

    def __init__(
        self,
        sentences_per_chunk: int = 5,
        overlap:             int = 1,
    ) -> None:
        self.sentences_per_chunk = sentences_per_chunk
        self.overlap             = overlap

    def chunk(self, text: str, source_id: str = "") -> List[dict]:
        if not text or not text.strip():
            return []

        sentences = [s.strip() for s in _SENT_RE.split(text) if s.strip()]
        if not sentences:
            return []

        chunks: List[dict] = []
        step   = max(1, self.sentences_per_chunk - self.overlap)
        idx    = 0
        chunk_n = 0

        while idx < len(sentences):
            window = sentences[idx: idx + self.sentences_per_chunk]
            chunks.append({
                "chunk_id":  f"{source_id}_sent_{chunk_n}",
                "source_id": source_id,
                "text":      " ".join(window),
                "index":     chunk_n,
            })
            idx    += step
            chunk_n += 1

        return chunks
