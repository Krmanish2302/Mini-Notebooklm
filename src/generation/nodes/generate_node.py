"""
generate_node.py — LangGraph node: invoke LLM (blocking or streaming).
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
            # Collect streamed tokens into a single string
            tokens = []
            for chunk in llm.stream([HumanMessage(content=prompt)]):
                tokens.append(chunk.content)
            raw_output = "".join(tokens)
        else:
            response   = llm.invoke([HumanMessage(content=prompt)])
            raw_output = response.content

        logger.info("[generate_response] Generated %d chars", len(raw_output))
        return {"raw_output": raw_output}
    except Exception as exc:
        logger.exception("[generate_response] Failed: %s", exc)
        return {"error": str(exc), "failed_node": "generate_response"}