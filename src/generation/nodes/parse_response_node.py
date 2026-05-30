"""
parse_response_node.py — LangGraph node: parse raw LLM output into clean answer + follow-ups.
"""
from __future__ import annotations
import logging

logger = logging.getLogger(__name__)


def parse_response(state: dict) -> dict:
    try:
        raw_output = state.get("raw_output", "")
        query      = state.get("query", "")

        from src.generation.response_parser import ResponseParser
        parsed = ResponseParser.parse(raw_output)

        logger.info("[parse_response] Answer len=%d follow_ups=%d",
                    len(parsed.answer), len(parsed.follow_ups))
        return {
            "answer":     parsed.answer,
            "follow_ups": parsed.follow_ups,
        }
    except Exception as exc:
        logger.exception("[parse_response] Failed: %s", exc)
        return {"error": str(exc), "failed_node": "parse_response"}