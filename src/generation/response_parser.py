"""
response_parser.py — Structured output parsing using LangChain output parsers.

Provides:
  ResponseParser   — parses raw LLM text into a GeneratedResponse Pydantic model
  FollowUpParser   — extracts follow-up questions from an answer string
"""
from __future__ import annotations
import re
from typing import List, Optional

from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class GeneratedResponse(BaseModel):
    """Structured output schema for LLM-generated answers."""
    answer:      str        = Field(description="Clean answer text with inline [S1]… citations")
    follow_ups:  List[str]  = Field(default_factory=list, description="2-3 follow-up questions")
    confidence:  Optional[float] = Field(default=None, description="Self-reported confidence 0-1")


class FollowUpList(BaseModel):
    """Schema for extracted follow-up questions."""
    questions: List[str] = Field(description="List of follow-up questions")


# ── Parsers ───────────────────────────────────────────────────────────────────

_str_parser     = StrOutputParser()
_response_parser = PydanticOutputParser(pydantic_object=GeneratedResponse)
_followup_parser = PydanticOutputParser(pydantic_object=FollowUpList)

_STRIP_PREFIXES = re.compile(
    r"^(?:A|ANSWER|DETAILED\s+ANSWER|EXPLAIN|RESPONSE)\s*:\s*",
    re.IGNORECASE,
)
_FOLLOWUP_BLOCK = re.compile(
    r"(?:follow[- ]?up|suggested|you\s+might\s+also\s+ask)[:\s\n]+(.*?)(?:\n\n|$)",
    re.IGNORECASE | re.DOTALL,
)
_CITE_PATTERN = re.compile(r"\[([Ss]\d{1,2})\]")


class ResponseParser:
    """
    Parses raw LLM text into a clean answer + metadata.

    Tries PydanticOutputParser first (structured JSON output from the LLM).
    Falls back to regex-based extraction if the LLM returned plain text.
    """

    @staticmethod
    def parse(raw: str) -> GeneratedResponse:
        """
        Parse raw LLM output → GeneratedResponse.
        Falls back gracefully to regex extraction.
        """
        # Try structured JSON parse first
        try:
            return _response_parser.parse(raw)
        except Exception:
            pass

        # Regex fallback
        cleaned    = re.sub(r"```[\s\S]*?```", "", raw)
        answer     = _STRIP_PREFIXES.sub("", cleaned.strip()).strip()
        # Normalize citations
        answer     = _CITE_PATTERN.sub(lambda m: f"[{m.group(1).upper()}]", answer)
        # Extract follow-ups
        follow_ups = FollowUpParser.extract(answer)
        answer     = _FOLLOWUP_BLOCK.sub("", answer).strip()
        return GeneratedResponse(answer=answer, follow_ups=follow_ups)


class FollowUpParser:
    """Extracts follow-up questions from an answer string."""

    @staticmethod
    def extract(text: str, max_questions: int = 3) -> List[str]:
        match = _FOLLOWUP_BLOCK.search(text)
        if not match:
            return []
        block = match.group(1)
        lines = re.findall(r"^\s*[-•*\d.)]+\s*(.+)$", block, re.MULTILINE)
        return [
            (l.strip().rstrip(".") + "?") if not l.strip().endswith("?") else l.strip()
            for l in lines if len(l.strip()) > 8
        ][:max_questions]