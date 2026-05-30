"""
rerank_node.py — LangGraph node: rerank retrieved documents.
"""
from __future__ import annotations
import logging

logger = logging.getLogger(__name__)


def rerank_docs(state: dict) -> dict:
    try:
        docs  = state.get("documents", [])
        query = state.get("query", "")
        top_k = state.get("top_k", 5)

        from src.retrieval.reranker import Reranker
        reranked = Reranker().rerank(query, docs, top_n=top_k)
        logger.info("[rerank_docs] %d → %d docs after reranking", len(docs), len(reranked))
        return {"reranked_docs": reranked}
    except Exception as exc:
        logger.exception("[rerank_docs] Failed: %s", exc)
        return {"error": str(exc), "failed_node": "rerank_docs"}