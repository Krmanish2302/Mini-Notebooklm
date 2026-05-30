"""
build_prompt_node.py — LangGraph node: build the LLM prompt.
"""
from __future__ import annotations
import logging

logger = logging.getLogger(__name__)


def build_prompt(state: dict) -> dict:
    try:
        query     = state.get("query", "")
        documents = state.get("documents", [])
        mode      = state.get("mode", "chat")
        history   = state.get("history", "")
        persona   = state.get("persona")

        if not query:
            return {"error": "No query provided", "failed_node": "build_prompt"}

        from src.generation.prompt_builder import PromptBuilder

        if mode == "study":
            prompt = PromptBuilder.build_study_prompt(query, documents, history)
        elif mode == "research":
            prompt = PromptBuilder.build_research_prompt(query, documents, history)
        else:
            prompt = PromptBuilder.build_chat_prompt(query, documents, history, persona)

        logger.info("[build_prompt] mode=%s prompt_len=%d", mode, len(prompt))
        return {"prompt": prompt}
    except Exception as exc:
        logger.exception("[build_prompt] Failed: %s", exc)
        return {"error": str(exc), "failed_node": "build_prompt"}
