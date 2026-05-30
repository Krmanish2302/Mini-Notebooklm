"""
citation_extractor.py — Extract and validate inline citations from LLM answers.

Uses LangChain output parsers + regex to extract structured citation metadata.
"""
from __future__ import annotations
import re
from typing import Any, Dict, List, Optional

from langchain_core.output_parsers import StrOutputParser

_CITE_RE   = re.compile(r"\[S(\d{1,2})\]", re.IGNORECASE)
_str_parser = StrOutputParser()


class CitationExtractor:
    """
    Extracts citation references ([S1], [S2]…) from an LLM answer and
    resolves them against the context chunks that were fed to the model.

    Usage:
        extractor = CitationExtractor(context_chunks)
        result    = extractor.extract(answer_text)
        # result → List[{"label": "S1", "source_id": "...", "snippet": "..."}]
    """

    def __init__(self, context_chunks: Optional[List[Dict[str, Any]]] = None):
        self.context_chunks = context_chunks or []
        self._label_map: Dict[str, Dict] = {}
        for i, chunk in enumerate(self.context_chunks, 1):
            label = f"S{i}"
            self._label_map[label] = chunk

    def extract(self, answer: str) -> List[Dict[str, Any]]:
        """
        Extract all inline citations and resolve them to source metadata.

        Returns:
            List of citation dicts: {label, source_id, source_name, snippet, page}
        """
        labels = list(dict.fromkeys(
            m.group(1).upper() for m in _CITE_RE.finditer(answer)
        ))
        results = []
        for label in labels:
            chunk = self._label_map.get(label, {})
            if hasattr(chunk, "page_content"):
                content = chunk.page_content
                meta    = chunk.metadata or {}
            else:
                content = chunk.get("content", "")
                meta    = {k: v for k, v in chunk.items() if k not in ("content",)}

            results.append({
                "label":       f"[S{label}]",
                "source_id":   meta.get("source_id", ""),
                "source_name": meta.get("source", meta.get("source_name", "")),
                "page":        meta.get("page", ""),
                "snippet":     content[:200],
            })
        return results

    def validate(self, answer: str) -> Dict[str, List[str]]:
        """
        Check for hallucinated or missing citations.

        Returns:
            {
                "valid":        List[str],  # cited AND present in context
                "hallucinated": List[str],  # cited but NOT in context
                "uncited":      List[str],  # in context but NOT cited
            }
        """
        cited_labels = {
            m.group(1).upper() for m in _CITE_RE.finditer(answer)
        }
        all_labels   = set(self._label_map.keys())
        return {
            "valid":        sorted(cited_labels & all_labels),
            "hallucinated": sorted(cited_labels - all_labels),
            "uncited":      sorted(all_labels - cited_labels),
        }