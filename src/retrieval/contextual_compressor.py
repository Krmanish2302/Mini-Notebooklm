"""
ContextualCompressor  —  used in Deep Research Mode (Step 4).

For each retrieved chunk, asks the LLM to extract ONLY the sentences
directly relevant to the query.  Chunks the LLM deems irrelevant are
dropped entirely (return value: None).

Token savings
-------------
Typically reduces total chunk tokens by 40-60% before the reranker pass,
which more than offsets the LLM call cost.

Behaviour
---------
* Chunks shorter than min_tokens pass through unchanged.
* Falls back to original chunk if the LLM call fails.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

_IRRELEVANT_MARKER = "\u2205"


class ContextualCompressor:
    """
    Parameters
    ----------
    llm        : callable(prompt: str) -> str
    min_tokens : int   chunks below this word count pass through unchanged (default 30)
    max_tokens : int   soft cap on output length (default 200 words)
    """

    def __init__(
        self, llm: Callable[[str], str], min_tokens: int = 30, max_tokens: int = 200
    ):
        self.llm = llm
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens

    def compress(
        self, chunks: List[Dict[str, Any]], query: str
    ) -> List[Dict[str, Any]]:
        """Compress each chunk; drop irrelevant ones. Order preserved."""
        return [r for r in (self._compress_one(c, query) for c in chunks) if r is not None]

    def _compress_one(
        self, chunk: Dict[str, Any], query: str
    ) -> Optional[Dict[str, Any]]:
        content = chunk.get("content", "")
        if len(content.split()) < self.min_tokens:
            return chunk
        prompt = (
            f"Given the QUESTION below, extract from the PASSAGE the sentences "
            f"that directly answer or are relevant to the question. "
            f"Output ONLY those sentences (max {self.max_tokens} words). "
            f"If NO sentence is relevant, output exactly: {_IRRELEVANT_MARKER}\n\n"
            f"QUESTION: {query}\n\nPASSAGE:\n{content}\n\nRELEVANT SENTENCES:"
        )
        try:
            compressed = self.llm(prompt).strip()
        except Exception as exc:
            logger.warning("ContextualCompressor LLM call failed: %s", exc)
            return chunk
        if compressed == _IRRELEVANT_MARKER or not compressed:
            logger.debug("ContextualCompressor: dropped chunk %s", chunk.get("id"))
            return None
        result = dict(chunk)
        result["content"] = compressed
        result["_original_content"] = content
        result["_compressed"] = True
        return result
