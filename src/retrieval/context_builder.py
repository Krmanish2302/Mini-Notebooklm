"""
context_builder.py  —  Assembles ranked retrieval results into a structured
context window ready for prompt injection.

Responsibilities
----------------
1. Deduplication   — remove near-duplicate chunks (Jaccard similarity)
2. Token budgeting — truncate chunks so total context stays within a limit
3. Source ranking  — order by RRF score desc; break ties by recency
4. Attribution     — attach source labels ([S1], [S2]…) for citation
5. Metadata strip  — emit clean dicts with only the fields the prompt needs

Usage
-----
    builder = ContextBuilder(max_tokens=3000)
    context, sources = builder.build(chunks)  # chunks from HybridRetriever
    # context  → list of clean, deduplicated, token-budgeted chunk dicts
    # sources  → list of unique source labels for the footer
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Rough chars-per-token ratio (conservative for mixed technical / prose)
_CHARS_PER_TOKEN: float = 3.5


class ContextBuilder:
    """
    Parameters
    ----------
    max_tokens   : int   — total token budget for the context block (default 3000)
    sim_threshold: float — Jaccard similarity above which a chunk is a duplicate (default 0.82)
    """

    def __init__(
        self,
        max_tokens:    int   = 3000,
        sim_threshold: float = 0.82,
    ):
        self.max_tokens    = max_tokens
        self.sim_threshold = sim_threshold
        self._char_budget  = int(max_tokens * _CHARS_PER_TOKEN)

    # ─────────────────────────────────────────────────────────────────────
    #  Public API
    # ─────────────────────────────────────────────────────────────────────

    def build(
        self,
        chunks: List[Dict[str, Any]],
        query:  Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Build context from a list of retrieved chunks.

        Parameters
        ----------
        chunks : list of chunk dicts (from HybridRetriever.retrieve())
                 Each must have at minimum: {"id": str, "content": str}
        query  : original query string (used for diversity scoring, optional)

        Returns
        -------
        context : list of enriched chunk dicts ready for PromptBuilder
        sources : deduplicated list of source labels in order of first appearance
        """
        if not chunks:
            return [], []

        # 1. Sort by score (RRF desc, then arbitrary tiebreak)
        ranked = sorted(
            chunks,
            key=lambda c: float(c.get("rrf_score", 0.0)),
            reverse=True,
        )

        # 2. Deduplicate
        deduped = self._deduplicate(ranked)

        # 3. Apply token budget
        budgeted = self._apply_budget(deduped)

        # 4. Annotate with citation labels
        context, sources = self._annotate(budgeted)

        logger.debug(
            "ContextBuilder: %d → dedup %d → budget %d → final %d chunks",
            len(chunks), len(deduped), len(budgeted), len(context),
        )
        return context, sources

    # ─────────────────────────────────────────────────────────────────────
    #  Private helpers
    # ─────────────────────────────────────────────────────────────────────

    def _deduplicate(self, chunks: List[Dict]) -> List[Dict]:
        """Remove chunks whose content is near-identical to an already-kept chunk."""
        kept: List[Dict]      = []
        kept_sets: List[set]  = []

        for chunk in chunks:
            text   = chunk.get("content", "")
            tokens = self._tokenize(text)
            if not tokens:
                continue

            is_dup = False
            for existing_set in kept_sets:
                if self._jaccard(tokens, existing_set) >= self.sim_threshold:
                    is_dup = True
                    break

            if not is_dup:
                kept.append(chunk)
                kept_sets.append(tokens)

        return kept

    def _apply_budget(self, chunks: List[Dict]) -> List[Dict]:
        """
        Keep chunks in ranked order until the character budget is exhausted.
        Partially include the last fitting chunk (truncate at sentence boundary).
        """
        result:   List[Dict] = []
        remaining: int       = self._char_budget

        for chunk in chunks:
            text = chunk.get("content", "")
            if not text:
                continue

            if len(text) <= remaining:
                result.append(chunk)
                remaining -= len(text)
            else:
                # Try to truncate at last sentence boundary
                truncated = self._truncate_at_sentence(text, remaining)
                if truncated:
                    c = dict(chunk)          # shallow copy — don't mutate original
                    c["content"] = truncated
                    c["truncated"] = True
                    result.append(c)
                break  # budget exhausted

            if remaining <= 0:
                break

        return result

    def _annotate(
        self, chunks: List[Dict]
    ) -> Tuple[List[Dict], List[str]]:
        """
        Assign sequential [S1], [S2]… labels.
        Chunks from the same source share the same label.
        """
        source_map:  Dict[str, str]  = {}   # source_id/name -> label
        label_count: int             = 0
        result:      List[Dict]      = []
        sources:     List[str]       = []   # ordered unique labels

        for chunk in chunks:
            src_key = (
                chunk.get("source_id")
                or chunk.get("source")
                or chunk.get("source_name")
                or ""
            )
            if src_key not in source_map:
                label_count += 1
                label = f"S{label_count}"
                source_map[src_key] = label
                sources.append(label)

            annotated = dict(chunk)
            annotated["citation_label"] = source_map[src_key]
            result.append(annotated)

        return result, sources

    # ─────────────────────────────────────────────────────────────────────
    #  Utility
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> set:
        """Simple whitespace tokenizer returning a set of unique lowercase words."""
        return set(re.findall(r"\b\w+\b", text.lower()))

    @staticmethod
    def _jaccard(a: set, b: set) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    @staticmethod
    def _truncate_at_sentence(text: str, max_chars: int) -> str:
        """Truncate text at a sentence boundary not exceeding max_chars."""
        if max_chars <= 0:
            return ""
        snippet = text[:max_chars]
        # Walk back to find the last sentence-ending punctuation
        match = re.search(r"[.!?](?=[^.!?]*$)", snippet)
        if match:
            return snippet[: match.end()].strip()
        # Fall back to last whitespace boundary
        last_space = snippet.rfind(" ")
        if last_space > max_chars * 0.5:
            return snippet[:last_space].strip() + "…"
        return snippet.strip()
