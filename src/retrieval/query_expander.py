"""
query_expander.py  —  LLM-assisted query expansion for Deep Research mode.

Provides two strategies:
  1. SubQueryDecomposer  — breaks a complex question into N focused sub-queries
                           using a lightweight LLM call.
  2. MultiQueryExpander  — generates alternative phrasings of the same question
                           to improve recall via union of results.

Both fall back gracefully when no LLM is available (returns original query).
"""
from __future__ import annotations

import logging
import re
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Sub-query decomposer
# ─────────────────────────────────────────────────────────────────────────────

class SubQueryDecomposer:
    """
    Decomposes a complex query into N atomic sub-queries via an LLM call.

    If the LLM is unavailable or returns garbage, falls back to returning
    the original query as a single-element list.

    Parameters
    ----------
    llm : callable  (str) -> str   — synchronous LLM invoke function
    n   : int                      — target number of sub-queries (default 3)
    """

    _PROMPT = """\
Break the following research question into {n} focused, self-contained sub-questions.
Each sub-question must be answerable on its own from a document corpus.
Return ONLY a numbered list — no preamble, no explanation.

Question: {query}
"""

    def __init__(self, llm: Optional[Callable[[str], str]] = None, n: int = 3):
        self.llm = llm
        self.n   = n

    def decompose(self, query: str) -> List[str]:
        """Return a list of sub-queries. Always includes the original as fallback."""
        if not self.llm:
            return [query]

        prompt = self._PROMPT.format(n=self.n, query=query)
        try:
            raw = self.llm(prompt)
            lines = re.findall(r"^\s*\d+[.)\-\s]+(.+)$", raw, re.MULTILINE)
            sub_queries = [l.strip() for l in lines if l.strip()]
            if len(sub_queries) >= 2:
                # Always keep the original query in the pool
                if query not in sub_queries:
                    sub_queries.insert(0, query)
                return sub_queries[: self.n + 1]
        except Exception as exc:
            logger.warning("SubQueryDecomposer.decompose failed: %s", exc)

        return [query]


# ─────────────────────────────────────────────────────────────────────────────
#  Multi-query expander
# ─────────────────────────────────────────────────────────────────────────────

class MultiQueryExpander:
    """
    Generates N alternative phrasings of the same question so that union
    retrieval improves recall against differently-worded source chunks.

    Useful for Chat mode when the user's query is ambiguous or very short.

    Parameters
    ----------
    llm : callable  (str) -> str
    n   : int       number of alternative phrasings (default 2)
    """

    _PROMPT = """\
Rephrase the following question in {n} different ways that preserve the same intent
but use different vocabulary and structure.
Return ONLY a numbered list — no preamble, no explanation.

Question: {query}
"""

    def __init__(self, llm: Optional[Callable[[str], str]] = None, n: int = 2):
        self.llm = llm
        self.n   = n

    def expand(self, query: str) -> List[str]:
        """Return original query + rephrased alternatives."""
        if not self.llm:
            return [query]

        prompt = self._PROMPT.format(n=self.n, query=query)
        try:
            raw = self.llm(prompt)
            lines = re.findall(r"^\s*\d+[.)\-\s]+(.+)$", raw, re.MULTILINE)
            alternatives = [l.strip() for l in lines if l.strip()]
            if alternatives:
                result = [query] + alternatives[: self.n]
                return result
        except Exception as exc:
            logger.warning("MultiQueryExpander.expand failed: %s", exc)

        return [query]
