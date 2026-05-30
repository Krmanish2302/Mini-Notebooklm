"""
generation_graph.py

LangGraph StateGraph for the full generation pipeline.

Flow:
  build_prompt
      │
      ▼
  generate_response
      │
      ▼
  parse_response
      │
      ▼
  extract_citations
      │
      ▼
     END

Any node failure → handle_error → END
"""
from __future__ import annotations
import logging
from langgraph.graph import StateGraph, END

from .state import GenerationState
from .nodes.build_prompt_node     import build_prompt
from .nodes.generate_node         import generate_response
from .nodes.parse_response_node   import parse_response
from .nodes.extract_citations_node import extract_citations, handle_error

logger = logging.getLogger(__name__)


def _check_error(s: dict) -> str:
    return "handle_error" if s.get("error") else "next"


def build_generation_graph() -> StateGraph:
    wf = StateGraph(GenerationState)

    wf.add_node("build_prompt",        build_prompt)
    wf.add_node("generate_response",   generate_response)
    wf.add_node("parse_response",      parse_response)
    wf.add_node("extract_citations",   extract_citations)
    wf.add_node("handle_error",        handle_error)

    wf.set_entry_point("build_prompt")

    wf.add_conditional_edges(
        "build_prompt",
        lambda s: "handle_error" if s.get("error") else "generate_response",
        {"generate_response": "generate_response", "handle_error": "handle_error"},
    )
    wf.add_conditional_edges(
        "generate_response",
        lambda s: "handle_error" if s.get("error") else "parse_response",
        {"parse_response": "parse_response", "handle_error": "handle_error"},
    )
    wf.add_conditional_edges(
        "parse_response",
        lambda s: "handle_error" if s.get("error") else "extract_citations",
        {"extract_citations": "extract_citations", "handle_error": "handle_error"},
    )
    wf.add_conditional_edges(
        "extract_citations",
        lambda s: "handle_error" if s.get("error") else END,
        {END: END, "handle_error": "handle_error"},
    )
    wf.add_edge("handle_error", END)

    return wf


generation_app = build_generation_graph().compile()