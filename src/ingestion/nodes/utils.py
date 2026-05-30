"""
utils.py

Shared utilities for ingestion nodes.

@safe_node(name)
----------------
Decorator that wraps any LangGraph node function in a try/except.
On exception it sets state["error"] and state["failed_node"] so
the graph can route to handle_error without crashing.

Usage:
    @safe_node("my_node")
    def my_node(state: dict) -> dict:
        ...  # any exception here is caught safely
"""
from __future__ import annotations

import functools
import logging
from typing import Callable

logger = logging.getLogger(__name__)


def safe_node(node_name: str) -> Callable:
    """Decorator: catch all exceptions, set error keys in state."""
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(state: dict) -> dict:
            try:
                return fn(state)
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "[%s] Unhandled exception: %s", node_name, exc
                )
                return {
                    "error":       str(exc),
                    "failed_node": node_name,
                }
        return wrapper
    return decorator
