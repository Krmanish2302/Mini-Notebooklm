"""
error_node.py — terminal error handler node for LangGraph.
"""
import logging
logger = logging.getLogger(__name__)


def handle_error(state: dict) -> dict:
    logger.error(
        "[handle_error] Pipeline failed at node='%s' source='%s': %s",
        state.get("failed_node", "unknown"),
        state.get("source_id",   "unknown"),
        state.get("error",       "Unknown error"),
    )
    return {}