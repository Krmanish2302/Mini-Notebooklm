"""
citation_extractor.py — Extract and validate inline citations from LLM answers.
"""
from __future__ import annotations
import re
from typing import Any, Dict, List, Optional

from langchain_core.output_parsers import StrOutputParser

_CITE_RE    = re.compile(r"\[S(\d{1,2})\]", re.IGNORECASE)
_str_parser = StrOutputParser()


class CitationExtractor:
    """
    Extracts [S1], [S2]… from an answer and resolves to source metadata.
    """

    def __init__(self, context_chunks: Optional[List[Dict[str, Any]]] = None):
        self.context_chunks = context_chunks or []
        self._label_map: Dict[str, Dict] = {}
        for i, chunk in enumerate(self.context_chunks, 1):
            self._label_map[f"S{i}"] = chunk

    def extract(self, answer: str) -> List[Dict[str, Any]]:
        labels = list(dict.fromkeys(
            m.group(1).upper() for m in _CITE_RE.finditer(answer)
        ))
        results = []
        for label in labels:
            chunk = self._label_map.get(f"S{label}", {})
            if hasattr(chunk, "page_content"):
                content = chunk.page_content
                meta    = chunk.metadata or {}
            else:
                content = chunk.get("content", "")
                meta    = {k: v for k, v in chunk.items() if k not in ("content",)}
            results.append({
                "label":       f"[S{label}]",
                "source_id":   meta.get("source_id", ""),
                "source_name": meta.get("source", meta.get("source_name", meta.get("source_id", ""))),
                "page":        meta.get("page", meta.get("page_number", "")),
                "snippet":     content[:200],
                "content":     content,
            })
        return results

    def validate(self, answer: str) -> Dict[str, List[str]]:
        cited_labels = {m.group(1).upper() for m in _CITE_RE.finditer(answer)}
        all_labels   = set(self._label_map.keys())
        return {
            "valid":        sorted(cited_labels & all_labels),
            "hallucinated": sorted(cited_labels - all_labels),
            "uncited":      sorted(all_labels - cited_labels),
        }
