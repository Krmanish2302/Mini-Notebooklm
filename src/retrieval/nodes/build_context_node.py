"""
build_context_node.py — LangGraph node: build final context string + error handler.

Fix #11: handle_error returns safe defaults so callers never KeyError on
         context / documents after a retrieval failure.
"""
from __future__ import annotations
import logging

logger = logging.getLogger(__name__)


def build_context(state: dict) -> dict:
    try:
        docs = (
            state.get("compressed_docs")
            or state.get("reranked_docs")
            or state.get("documents", [])
        )
        query = state.get("query", "")

        from src.retrieval.context_builder import ContextBuilder
        context = ContextBuilder().build(docs, query)

        logger.info("[build_context] Context built from %d docs (%d chars)", len(docs), len(context))
        return {
            "context":   context,
            "documents": docs,
            "metadata":  {"num_docs": len(docs), "context_chars": len(context)},
        }
    except Exception as exc:
        logger.exception("[build_context] Failed: %s", exc)
        return {"error": str(exc), "failed_node": "build_context"}


def handle_error(state: dict) -> dict:
    logger.error(
        "[retrieval] FAILED at node='%s': %s",
        state.get("failed_node", "unknown"),
        state.get("error", "Unknown error"),
    )
    # FIX #11: return safe defaults — previously returned {} which left
    # context / documents unset, causing KeyError in master_pipeline.py
    return {
        "context":   "",
        "documents": [],
        "metadata":  {"num_docs": 0, "context_chars": 0},
    }
