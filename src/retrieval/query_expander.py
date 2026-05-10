"""
query_expander.py  —  Query expansion utilities for retrieval augmentation.

Bug fixes applied (2026-05-10 audit):
  BUG-013: SubQueryDecomposer fallback now returns 3 lexical variants instead
           of a single-element list containing the original query unchanged.
           Previously: silent no-op when LLM was unavailable.
           Now:  3 meaningful variants (original + 2 paraphrases via simple
                 heuristic transforms) so retrieval still improves.

Classes
-------
SubQueryDecomposer  — break complex queries into N focused sub-queries (LLM)
MultiQueryExpander  — generate alternative phrasings for broader recall
"""
from __future__ import annotations

import logging
import re
from typing import Any, Callable, List, Optional

logger = logging.getLogger(__name__)

# ── heuristic fallback helpers ────────────────────────────────────────────────

def _heuristic_variants(query: str, n: int = 3) -> List[str]:
    """
    BUG-013 fallback: produce n simple lexical variants of a query so that
    retrieval is still broader than a single query when the LLM is unavailable.
    """
    variants = [query]

    # Variant 2: rephrase with "what is" / "explain" / "describe"
    stripped = query.rstrip("?.").strip()
    if not re.match(r"^(what|how|why|when|where|who|explain|describe|list)",
                    stripped.lower()):
        variants.append(f"What is {stripped}?")
    else:
        variants.append(f"Explain in detail: {stripped}")

    # Variant 3: keyword extraction (drop stop words, join remaining)
    _STOP = {"the","a","an","is","are","was","were","of","in","on",
             "at","to","for","and","or","but","with","about","what",
             "how","why","when","where","who"}
    keywords = " ".join(
        w for w in re.findall(r"\b\w+\b", stripped.lower()) if w not in _STOP
    )
    if keywords and keywords != stripped.lower():
        variants.append(keywords)
    elif len(variants) < n:
        variants.append(f"Summarise: {stripped}")

    return variants[:n]


# ─────────────────────────────────────────────────────────────────────────────

class SubQueryDecomposer:
    """
    Decomposes a complex research query into N focused, self-contained
    sub-queries using an LLM call.

    Falls back to heuristic variants (BUG-013 fix) when the LLM is
    unavailable or returns an unparseable response.

    Parameters
    ----------
    llm : callable (str) -> str   — LLM invoke callable, or None
    n   : int                     — number of sub-queries to generate
    """

    _SYSTEM = (
        "You are a research assistant. Break the following question into "
        "{n} focused, self-contained sub-questions that together fully cover "
        "the original question. Output ONLY a numbered list, one per line."
    )

    def __init__(
        self,
        llm: Optional[Callable[[str], str]] = None,
        n:   int = 3,
    ):
        self.llm = llm
        self.n   = n

    def decompose(self, query: str) -> List[str]:
        if not self.llm:
            logger.debug(
                "SubQueryDecomposer: no LLM — using heuristic fallback for '%s'", query
            )
            return _heuristic_variants(query, self.n)  # BUG-013 fix

        prompt = (
            self._SYSTEM.format(n=self.n)
            + f"\n\nQuestion: {query}\n\nSub-questions:"
        )
        try:
            raw = self.llm(prompt)
            sub_queries = self._parse(raw)
            if len(sub_queries) >= 2:
                return sub_queries[: self.n]
            # LLM returned fewer than 2 — fallback
            logger.warning(
                "SubQueryDecomposer: LLM returned %d sub-queries (expected %d) — "
                "augmenting with heuristic variants",
                len(sub_queries), self.n,
            )
            return (sub_queries + _heuristic_variants(query, self.n))[: self.n]
        except Exception as exc:
            logger.warning(
                "SubQueryDecomposer: LLM call failed (%s) — using heuristic fallback", exc
            )
            return _heuristic_variants(query, self.n)  # BUG-013 fix

    @staticmethod
    def _parse(raw: str) -> List[str]:
        lines = []
        for line in raw.splitlines():
            line = re.sub(r"^\s*\d+[.)\-]\s*", "", line).strip()
            if line and len(line) > 5:
                lines.append(line)
        return lines


class MultiQueryExpander:
    """
    Generates N alternative phrasings of the same question for broader
    retrieval recall.  Used in Chat mode for ambiguous queries.

    Falls back to heuristic variants when LLM is unavailable.
    """

    _SYSTEM = (
        "Generate {n} alternative phrasings of the following question that "
        "preserve its meaning but use different words. "
        "Output ONLY the phrasings, one per line, no numbering."
    )

    def __init__(
        self,
        llm: Optional[Callable[[str], str]] = None,
        n:   int = 3,
    ):
        self.llm = llm
        self.n   = n

    def expand(self, query: str) -> List[str]:
        if not self.llm:
            return _heuristic_variants(query, self.n)  # BUG-013 fix
        prompt = (
            self._SYSTEM.format(n=self.n)
            + f"\n\nOriginal question: {query}\n\nAlternative phrasings:"
        )
        try:
            raw      = self.llm(prompt)
            variants = [l.strip() for l in raw.splitlines() if l.strip()][: self.n]
            if not variants:
                return _heuristic_variants(query, self.n)
            # Always include the original
            if query not in variants:
                variants.insert(0, query)
            return variants[: self.n]
        except Exception as exc:
            logger.warning("MultiQueryExpander failed (%s) — fallback", exc)
            return _heuristic_variants(query, self.n)
