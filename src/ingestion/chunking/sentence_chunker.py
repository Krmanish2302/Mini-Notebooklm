"""sentence_chunker.py — splits preprocessed text into sentence-level chunks."""
from __future__ import annotations
import re, uuid
from typing import Any, Dict, List
from .base_chunker import BaseChunker

class SentenceChunker(BaseChunker):
    """
    Splits text into individual sentence chunks.
    Uses a simple regex sentence splitter (no NLTK dependency).
    Groups sentences into windows of `window_size` for context overlap.
    """
    def __init__(self, window_size: int = 3, **kwargs):
        self.window_size = window_size

    def chunk(self, preprocessed: Any, source_id: str = "") -> List[Dict]:
        # Accept either a string or preprocessed dict
        if isinstance(preprocessed, dict):
            text = preprocessed.get("text") or preprocessed.get("content", "")
        else:
            text = str(preprocessed)

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        for i in range(0, len(sentences), max(1, self.window_size - 1)):
            window = sentences[i : i + self.window_size]
            chunk_text = " ".join(window)
            chunks.append({
                "id":        str(uuid.uuid4()),
                "text":      chunk_text,
                "source_id": source_id,
                "metadata":  {
                    "chunker":        "sentence",
                    "sentence_start": i,
                    "sentence_end":   i + len(window) - 1,
                },
            })
        return chunks
