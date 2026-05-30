"""
expand_node.py — LangGraph node: expand query into sub-queries.
"""
from __future__ import annotations
import logging
import os

logger = logging.getLogger(__name__)
EXPAND_QUERIES = os.getenv("EXPAND_QUERIES", "true").lower() == "true"


def expand_query(state: dict) -> dict:
    try:
        query = state.get("query", "")
        if not query:
            return {"error": "No query provided", "failed_node": "expand_query"}

        # BUG-RET-06: state flag renamed do_expand (was expand_query)
        if not state.get("do_expand", EXPAND_QUERIES):
            logger.info("[expand_query] Expansion disabled — using original query")
            return {"expanded_queries": [query]}

        from src.retrieval.query_expander import SubQueryDecomposer
        queries = SubQueryDecomposer(n=3, use_llm=True).decompose(query)
        logger.info("[expand_query] %d queries generated", len(queries))
        return {"expanded_queries": queries}
    except Exception as exc:
        logger.exception("[expand_query] Failed: %s", exc)
        return {"error": str(exc), "failed_node": "expand_query"}
