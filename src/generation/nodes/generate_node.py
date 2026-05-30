"""
generate_node.py — LangGraph node: invoke LLM (blocking or streaming).

Fix #3: Guard chunk.content against None (Groq/Anthropic emit None-content
         chunks for tool-call and metadata events). Without the guard,
         "".join(tokens) raises TypeError and streaming always crashes.
"""
from __future__ import annotations
import logging

from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)


def generate_response(state: dict) -> dict:
    try:
        prompt = state.get("prompt", "")
        stream = state.get("stream", False)

        if not prompt:
            return {"error": "No prompt in state", "failed_node": "generate_response"}

        from src.generation.llm_registry import LLMRegistry
        llm = LLMRegistry.get()

        if stream:
            tokens = []
            for chunk in llm.stream([HumanMessage(content=prompt)]):
                # FIX #3: chunk.content can be None for tool/metadata events
                tokens.append(chunk.content or "")
            raw_output = "".join(tokens)
        else:
            response   = llm.invoke([HumanMessage(content=prompt)])
            raw_output = response.content or ""

        logger.info("[generate_response] Generated %d chars", len(raw_output))
        return {"raw_output": raw_output}
    except Exception as exc:
        logger.exception("[generate_response] Failed: %s", exc)
        return {"error": str(exc), "failed_node": "generate_response"}
