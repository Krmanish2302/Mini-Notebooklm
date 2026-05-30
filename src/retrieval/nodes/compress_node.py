"""
compress_node.py — LangGraph node: contextual compression (optional, costs LLM calls).
"""

from __future__ import annotations
import logging

logger = logging.getLogger(__name__)


def compress_docs(state: dict) -> dict:
    try:
        docs = state.get("reranked_docs") or state.get("documents", [])
        query = state.get("query", "")

        if not state.get("use_compression", False):
            logger.info("[compress_docs] Compression disabled — skipping")
            return {"compressed_docs": docs}

        from src.retrieval.contextual_compressor import ContextualCompressor

        compressed = ContextualCompressor().compress(query, docs)
        logger.info(
            "[compress_docs] %d → %d docs after compression", len(docs), len(compressed)
        )
        return {"compressed_docs": compressed}
    except Exception as exc:
        logger.exception("[compress_docs] Failed: %s", exc)
        return {"error": str(exc), "failed_node": "compress_docs"}
