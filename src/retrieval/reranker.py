"""
reranker.py

Cross-encoder reranker using Hugging Face CrossEncoder model:
cross-encoder/ms-marco-MiniLM-L-4-v2
"""
from __future__ import annotations
import logging
from typing import List

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# Module-level lazy-loaded singleton to avoid reloading from disk on every query
_cross_encoder: CrossEncoder | None = None

def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-4-v2")
        logger.info("[Reranker] CrossEncoder model 'cross-encoder/ms-marco-MiniLM-L-4-v2' loaded (singleton)")
    return _cross_encoder

class Reranker:
    """
    Reranks retrieved documents by relevance to the query using
    cross-encoder/ms-marco-MiniLM-L-4-v2.
    """

    def rerank(self, query: str, docs: List[Document], top_n: int = 5) -> List[Document]:
        if not docs:
            return docs

        try:
            encoder = _get_cross_encoder()
            pairs = [[query, doc.page_content] for doc in docs]
            scores = encoder.predict(pairs)
            for doc, score in zip(docs, scores):
                doc.metadata["relevance_score"] = float(score)
            ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
            logger.info("[Reranker] Cross-encoder reranked %d → %d docs", len(docs), min(len(docs), top_n))
            return [d for d, _ in ranked[:top_n]]
        except Exception as exc:
            logger.warning("[Reranker] Cross-encoder failed (%s) — score-sort fallback", exc)
            return sorted(
                docs,
                key=lambda d: d.metadata.get("relevance_score", d.metadata.get("score", 0.0)),
                reverse=True,
            )[:top_n]
