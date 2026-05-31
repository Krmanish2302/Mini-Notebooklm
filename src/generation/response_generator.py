"""
response_generator.py — Structured response assembly.

Takes raw LLM output and enriches into:
    {
        "answer":          str,
        "citations":       List[dict],
        "follow_ups":      List[str],
        "sources_used":    List[str],
        "chunks_used":     List[dict],
        "tokens_estimate": int,
    }
"""
from __future__ import annotations
import logging
import re
from typing import Any, Dict, List, Optional

from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

_CHARS_PER_TOKEN = 3.5
_CITE_PATTERN    = re.compile(r"\[([Ss]\d{1,2})\]")
_STRIP_PREFIXES  = re.compile(
    r"^(?:A|ANSWER|DETAILED\s+ANSWER|EXPLAIN|RESPONSE)\s*:\s*", re.IGNORECASE
)
_FOLLOWUP_BLOCK  = re.compile(
    r"(?:follow[- ]?up|suggested|you\s+might\s+also\s+ask)[:\s\n]+(.*?)(?:\n\n|$)",
    re.IGNORECASE | re.DOTALL,
)

_str_parser = StrOutputParser()


class ResponseGenerator:
    """
    Stateless response assembler. Call assemble() after every LLM invocation.
    """

    def __init__(self, context_chunks: Optional[List[Dict[str, Any]]] = None):
        self.context_chunks: List[Dict[str, Any]] = context_chunks or []

    def assemble(
        self,
        raw_llm_output: str,
        query:          str  = "",
        generate_follow_ups: bool = True,
    ) -> Dict[str, Any]:
        cleaned    = self._clean(raw_llm_output)
        answer     = self._strip_follow_up_block(cleaned)
        follow_ups = self._extract_follow_ups(cleaned, query, generate_follow_ups)
        answer     = _CITE_PATTERN.sub(lambda m: f"[{m.group(1).upper()}]", answer)
        used_labels = list(dict.fromkeys(_CITE_PATTERN.findall(answer)))
        citations   = self._build_citations(used_labels)
        chunks_used = [
            c for c in self.context_chunks
            if c.get("citation_label", "") in used_labels
        ]

        # Strip inline citation markers from final answer text
        clean_answer = _CITE_PATTERN.sub("", answer)
        clean_answer = re.sub(r"\s+([.,!?;:])", r"\1", clean_answer)
        clean_answer = re.sub(r" +", " ", clean_answer).strip()

        return {
            "answer":          clean_answer,
            "citations":       citations,
            "follow_ups":      follow_ups,
            "sources_used":    used_labels,
            "chunks_used":     chunks_used,
            "tokens_estimate": self._token_estimate(clean_answer),
        }

    @staticmethod
    def _clean(text: str) -> str:
        text = re.sub(r"```[\s\S]*?```", "", text)
        text = _STRIP_PREFIXES.sub("", text.strip())
        return text.strip()

    @staticmethod
    def _strip_follow_up_block(text: str) -> str:
        return _FOLLOWUP_BLOCK.sub("", text).strip()

    @staticmethod
    def _extract_follow_ups(text: str, query: str, enabled: bool) -> List[str]:
        if not enabled:
            return []
        match = _FOLLOWUP_BLOCK.search(text)
        if not match:
            return []
        block = match.group(1)
        lines = re.findall(r"^\s*[-•*\d.)]+\s*(.+)$", block, re.MULTILINE)
        return [
            (line.strip().rstrip(".") + "?") if not line.strip().endswith("?") else line.strip()
            for line in lines if len(line.strip()) > 8
        ][:3]

    def _build_citations(self, used_labels: List[str]) -> List[Dict[str, Any]]:
        label_map: Dict[str, Dict] = {}
        for chunk in self.context_chunks:
            lbl = chunk.get("citation_label", "")
            if lbl and lbl not in label_map:
                label_map[lbl] = chunk
        citations = []
        for lbl in used_labels:
            chunk = label_map.get(lbl)
            if chunk:
                citations.append({
                    "label":           lbl,
                    "source_id":       chunk.get("source_id", ""),
                    "source_name":     chunk.get("source_name", chunk.get("source", chunk.get("source_id", ""))),
                    "page":            chunk.get("page", chunk.get("page_number", "")),
                    "content_snippet": chunk.get("content", "")[:200],
                    "content":         chunk.get("content", ""),
                    "score":           chunk.get("rrf_score", 0.0),
                })
            else:
                citations.append({
                    "label": lbl, "source_id": "", "source_name": "", "page": "",
                    "content_snippet": "", "content": "", "score": 0.0,
                })
        return citations

    @staticmethod
    def _token_estimate(text: str) -> int:
        return max(1, int(len(text) / _CHARS_PER_TOKEN))
