"""
context_builder.py

Formats a List[Document] into a clean context string for the LLM.

Strategies:
  "numbered"   — [1] source | content (default)
  "markdown"   — ## Source\ncontent
  "plain"      — just content blocks separated by ---
  "cited"      — content with inline [source_id:page] citations

Usage:
    from src.retrieval.context_builder import ContextBuilder
    ctx = ContextBuilder().build(docs, query="What is X?")
"""
from __future__ import annotations
import os
from typing import List

from langchain_core.documents import Document

MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "12000"))
CONTEXT_STRATEGY  = os.getenv("CONTEXT_STRATEGY", "numbered")


class ContextBuilder:
    """
    Builds a formatted context string from retrieved Documents.

    Args:
        strategy:          "numbered" | "markdown" | "plain" | "cited"
        max_context_chars: Hard truncation limit for total context string.
    """

    def __init__(
        self,
        strategy:          str = CONTEXT_STRATEGY,
        max_context_chars: int = MAX_CONTEXT_CHARS,
    ):
        self.strategy          = strategy
        self.max_context_chars = max_context_chars

    def build(self, docs: List[Document], query: str = "") -> str:
        if not docs:
            return "No relevant context found."

        builders = {
            "numbered": self._numbered,
            "markdown":  self._markdown,
            "plain":     self._plain,
            "cited":     self._cited,
        }
        fn  = builders.get(self.strategy, self._numbered)
        ctx = fn(docs)
        return ctx[:self.max_context_chars]

    def _numbered(self, docs: List[Document]) -> str:
        parts = []
        for i, doc in enumerate(docs, 1):
            source  = doc.metadata.get("source_id", "unknown")
            page    = doc.metadata.get("page", "")
            loc     = f"{source}" + (f" p.{page}" if page else "")
            parts.append(f"[{i}] {loc}\n{doc.page_content.strip()}")
        return "\n\n".join(parts)

    def _markdown(self, docs: List[Document]) -> str:
        parts = []
        for doc in docs:
            source = doc.metadata.get("source_id", "unknown")
            parts.append(f"## {source}\n{doc.page_content.strip()}")
        return "\n\n".join(parts)

    def _plain(self, docs: List[Document]) -> str:
        return "\n\n---\n\n".join(d.page_content.strip() for d in docs)

    def _cited(self, docs: List[Document]) -> str:
        parts = []
        for doc in docs:
            source = doc.metadata.get("source_id", "unknown")
            page   = doc.metadata.get("page", "")
            tag    = f"[{source}:{page}]" if page else f"[{source}]"
            parts.append(f"{doc.page_content.strip()} {tag}")
        return "\n\n".join(parts)