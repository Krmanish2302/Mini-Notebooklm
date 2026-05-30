"""
content_analyzer.py

Analyzes Document content to detect type, language, and structure.
Returns metadata enrichments — no LLM calls.
"""
from __future__ import annotations
import re
from typing import Any, Dict, List
from langchain_core.documents import Document


class ContentAnalyzer:
    def analyze(self, docs: List[Document]) -> List[Dict[str, Any]]:
        return [self._analyze_one(doc) for doc in docs]

    def _analyze_one(self, doc: Document) -> Dict[str, Any]:
        text = doc.page_content
        return {
            "word_count":    len(text.split()),
            "char_count":    len(text),
            "has_tables":    bool(re.search(r"\|.+\|", text)),
            "has_code":      bool(re.search(r"```|def |class |import ", text)),
            "has_headings":  bool(re.search(r"^#{1,3}\s", text, re.MULTILINE)),
            "avg_word_len":  (sum(len(w) for w in text.split()) / max(len(text.split()), 1)),
            "source_id":     doc.metadata.get("source_id", ""),
        }