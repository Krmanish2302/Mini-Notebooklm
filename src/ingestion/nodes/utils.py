"""
utils.py — @safe_node decorator for all LangGraph ingestion nodes.

Catches any unhandled exception, sets state["error"] and state["failed_node"],
so the graph conditional edges can route to handle_error cleanly.
"""
from __future__ import annotations
import functools
import logging
from typing import Callable

logger = logging.getLogger(__name__)


def safe_node(node_name: str) -> Callable:
    """Wrap a node function — catch exceptions → set error keys in state."""
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(state: dict) -> dict:
            try:
                return fn(state)
            except Exception as exc:
                logger.exception("[%s] Unhandled exception: %s", node_name, exc)
                return {"error": str(exc), "failed_node": node_name}
        return wrapper
    return decorator