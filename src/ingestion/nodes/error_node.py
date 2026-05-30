"""
error_node.py

LangGraph node: centralised error handler.

Any node that catches an unexpected exception should set:
    state["error"]       = str(exception)
    state["failed_node"] = "<node_name>"

and return those keys.  The conditional edges in ingestion_graph.py
will route to this node, which logs the error and terminates the
pipeline gracefully rather than propagating an uncaught exception.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def handle_error(state: dict) -> dict:
    """
    LangGraph node — log error and terminate pipeline.

    Reads:  state["error"], state["failed_node"]
    Writes: nothing new (terminal node)
    """
    error       = state.get("error",       "Unknown error")
    failed_node = state.get("failed_node", "unknown")
    source_id   = state.get("source_id",   "unknown")

    logger.error(
        "[handle_error] Pipeline failed at node='%s' for source_id='%s': %s",
        failed_node, source_id, error,
    )
    # Return state unchanged — this is a terminal node
    return {}
