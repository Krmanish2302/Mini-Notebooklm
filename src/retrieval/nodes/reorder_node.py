"""
reorder_node.py — LangGraph node: lost-in-the-middle reordering.
"""
from __future__ import annotations
import logging

from src.retrieval.reorder import reorder_chunks

logger = logging.getLogger(__name__)


def reorder_docs(state: dict) -> dict:
    try:
        docs = state.get("reranked_docs") or state.get("documents", [])
        
        if not state.get("use_reordering", True):
            logger.info("[reorder_docs] Reordering disabled — skipping")
            return {"reordered_docs": docs}

        # Pair each document with its score from metadata
        chunks_with_scores = []
        for doc in docs:
            score = doc.metadata.get("relevance_score", doc.metadata.get("score", 0.0))
            # Handle string scores or missing scores by converting to float
            try:
                score = float(score)
            except (ValueError, TypeError):
                score = 0.0
            chunks_with_scores.append((doc, score))

        reordered = reorder_chunks(chunks_with_scores)
        logger.info(
            "[reorder_docs] Reordered %d docs using lost-in-the-middle", len(reordered)
        )
        return {"reordered_docs": reordered}
    except Exception as exc:
        logger.exception("[reorder_docs] Failed: %s", exc)
        return {"error": str(exc), "failed_node": "reorder_docs"}
