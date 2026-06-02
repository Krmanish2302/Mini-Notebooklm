"""
mode_router_node.py — LangGraph node: route queries based on retrieval mode.
"""
from __future__ import annotations
import logging

logger = logging.getLogger(__name__)

def mode_router(state: dict) -> str:
    """
    Routes the retrieval flow based on state["mode"].
    Returns the next node name.
    """
    mode = state.get("mode") or "chat"
    mode_clean = mode.strip().lower()
    
    if mode_clean in ("research", "deep_research", "deep"):
        route = "deep_research_retrieve"
    elif mode_clean == "study":
        route = "study_retrieve"
    else:
        route = "chat_retrieve"
        
    logger.info("[mode_router] mode='%s' -> next_node='%s'", mode, route)
    return route

