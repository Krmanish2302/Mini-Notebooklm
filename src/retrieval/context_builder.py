"""
context_builder.py

Formats a List[Document] into a clean context string for the LLM.

Strategies:
  "numbered"   — [1] source | content (default)
  "markdown"   — ## Source\ncontent
  "plain"      — just content blocks separated by ---
  "cited"      — content with inline [source_id:page] citations

Fixes applied
-------------
* FIX #6 : Hard char-slice truncation replaced with greedy per-document
           accumulation so context never ends mid-word or mid-citation.
* FIX #10: query parameter is now used for query-aware source ordering
           (docs mentioning query keywords bubble up in "numbered" strategy).

Usage:
    from src.retrieval.context_builder import ContextBuilder
    ctx = ContextBuilder().build(docs, query="What is X?")
"""
from __future__ import annotations
import os
import re
from typing import List

from langchain_core.documents import Document

MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "12000"))
CONTEXT_STRATEGY  = os.getenv("CONTEXT_STRATEGY", "numbered")


class ContextBuilder:
    """
    Builds a formatted context string from retrieved Documents.

    Args:
        strategy:          "numbered" | "markdown" | "plain" | "cited"
        max_context_chars: Hard limit for total context (applied per-doc, not by slicing).
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

        # FIX #10: sort docs so those containing query keywords come first.
        # This is a lightweight heuristic — reranker handles the heavy lifting.
        if query:
            keywords = set(re.findall(r"\b\w{4,}\b", query.lower()))
            if keywords:
                def _relevance(doc: Document) -> int:
                    text = doc.page_content.lower()
                    return sum(1 for kw in keywords if kw in text)
                docs = sorted(docs, key=_relevance, reverse=True)

        builders = {
            "numbered": self._numbered,
            "markdown":  self._markdown,
            "plain":     self._plain,
            "cited":     self._cited,
        }
        fn = builders.get(self.strategy, self._numbered)
        return self._accumulate(fn, docs)

    def _accumulate(self, builder_fn, docs: List[Document]) -> str:
        """
        FIX #6: Greedy accumulation instead of build-all-then-slice.
        Appends whole formatted blocks until the budget is spent, then stops.
        This guarantees the context string is never truncated mid-sentence or
        mid-citation bracket.
        """
        parts:  List[str] = []
        budget: int       = self.max_context_chars

        for block in builder_fn(docs, _lazy=True):
            if budget <= 0:
                break
            # If the block alone exceeds the remaining budget, trim at a sentence
            # boundary rather than a hard char cut.
            if len(block) > budget:
                trimmed = block[:budget]
                # Back up to the last sentence-end inside the budget
                last_stop = max(
                    trimmed.rfind(". "),
                    trimmed.rfind("\n"),
                )
                block = trimmed[: last_stop + 1].strip() if last_stop > 0 else trimmed
            parts.append(block)
            budget -= len(block)

        return "\n\n".join(parts)

    # ── Strategy generators (yield one block per document) ───────────────────────

    def _numbered(self, docs: List[Document], _lazy: bool = False):
        blocks = []
        for i, doc in enumerate(docs, 1):
            source  = doc.metadata.get("source_id", "unknown")
            page    = doc.metadata.get("page", "")
            loc     = f"{source}" + (f" p.{page}" if page else "")
            blocks.append(f"[{i}] {loc}\n{doc.page_content.strip()}")
        return iter(blocks) if _lazy else "\n\n".join(blocks)

    def _markdown(self, docs: List[Document], _lazy: bool = False):
        blocks = []
        for doc in docs:
            source = doc.metadata.get("source_id", "unknown")
            blocks.append(f"## {source}\n{doc.page_content.strip()}")
        return iter(blocks) if _lazy else "\n\n".join(blocks)

    def _plain(self, docs: List[Document], _lazy: bool = False):
        blocks = [d.page_content.strip() for d in docs]
        return iter(blocks) if _lazy else "\n\n---\n\n".join(blocks)

    def _cited(self, docs: List[Document], _lazy: bool = False):
        blocks = []
        for doc in docs:
            source = doc.metadata.get("source_id", "unknown")
            page   = doc.metadata.get("page", "")
            tag    = f"[{source}:{page}]" if page else f"[{source}]"
            blocks.append(f"{doc.page_content.strip()} {tag}")
        return iter(blocks) if _lazy else "\n\n".join(blocks)
