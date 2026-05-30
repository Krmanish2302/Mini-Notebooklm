"""
reranker.py

Cross-encoder reranker using LangChain's ContextualCompressionRetriever
with a CohereRerank or FlashrankRerank compressor.

Falls back to simple score-based reranking if no API key available.

Fix #8: FlashrankRerank is now a module-level singleton (lazy init) so the
        cross-encoder model is NOT reloaded from disk on every query call.
        Saves 200–400 ms per request.

Usage:
    from src.retrieval.reranker import Reranker
    reranker = Reranker()
    reranked = reranker.rerank(query="your query", docs=documents, top_n=5)
"""
from __future__ import annotations
import logging
import os
from typing import List, Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

RERANK_PROVIDER = os.getenv("RERANK_PROVIDER", "flashrank")  # "cohere" | "flashrank" | "none"
COHERE_API_KEY  = os.getenv("COHERE_API_KEY", "")

# FIX #8: module-level singletons — avoid reloading the model on every query
_flashrank_compressor: Optional[object] = None
_cohere_compressor:   Optional[object] = None


def _get_flashrank_compressor(top_n: int):
    """Lazy-init FlashrankRerank singleton (model loaded once per process)."""
    global _flashrank_compressor
    if _flashrank_compressor is None:
        from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
        _flashrank_compressor = FlashrankRerank(top_n=top_n)
        logger.info("[Reranker] FlashrankRerank model loaded (singleton)")
    else:
        # update top_n for this call without reloading the model
        _flashrank_compressor.top_n = top_n
    return _flashrank_compressor


def _get_cohere_compressor(top_n: int):
    """Lazy-init CohereRerank singleton."""
    global _cohere_compressor
    if _cohere_compressor is None:
        from langchain_cohere import CohereRerank
        _cohere_compressor = CohereRerank(
            cohere_api_key=COHERE_API_KEY,
            top_n=top_n,
            model="rerank-english-v3.0",
        )
        logger.info("[Reranker] CohereRerank singleton created")
    else:
        _cohere_compressor.top_n = top_n
    return _cohere_compressor


class Reranker:
    """
    Reranks retrieved documents by relevance to the query.

    Providers (set RERANK_PROVIDER env var):
      flashrank  — local, free, no API key (default)
      cohere     — Cohere Rerank API (requires COHERE_API_KEY)
      none       — pass-through (no reranking)
    """

    def rerank(self, query: str, docs: List[Document], top_n: int = 5) -> List[Document]:
        if not docs:
            return docs

        if RERANK_PROVIDER == "cohere" and COHERE_API_KEY:
            return self._cohere_rerank(query, docs, top_n)
        elif RERANK_PROVIDER == "flashrank":
            return self._flashrank_rerank(query, docs, top_n)
        else:
            logger.info("[Reranker] provider='none' — returning docs as-is")
            return docs[:top_n]

    def _cohere_rerank(self, query: str, docs: List[Document], top_n: int) -> List[Document]:
        try:
            compressor = _get_cohere_compressor(top_n)
            compressed = compressor.compress_documents(docs, query)
            logger.info("[Reranker] Cohere reranked %d → %d docs", len(docs), len(compressed))
            return compressed
        except Exception as exc:
            logger.warning("[Reranker] Cohere failed (%s) — flashrank fallback", exc)
            return self._flashrank_rerank(query, docs, top_n)

    def _flashrank_rerank(self, query: str, docs: List[Document], top_n: int) -> List[Document]:
        try:
            compressor = _get_flashrank_compressor(top_n)
            compressed = compressor.compress_documents(docs, query)
            logger.info("[Reranker] Flashrank reranked %d → %d docs", len(docs), len(compressed))
            return compressed
        except Exception as exc:
            logger.warning("[Reranker] Flashrank failed (%s) — score-sort fallback", exc)
            return sorted(
                docs,
                key=lambda d: d.metadata.get("relevance_score", d.metadata.get("score", 0)),
                reverse=True,
            )[:top_n]
