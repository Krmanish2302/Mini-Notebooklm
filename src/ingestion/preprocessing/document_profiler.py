"""
document_profiler.py

Profiles a list of Documents and returns a structured summary dict.
Used by master_pipeline.py to log ingestion stats.
No LLM calls.
"""
from __future__ import annotations
from typing import Any, Dict, List
from langchain_core.documents import Document


class DocumentProfiler:
    def profile(self, docs: List[Document]) -> Dict[str, Any]:
        if not docs:
            return {"total_docs": 0, "total_words": 0, "avg_words": 0, "source_types": []}

        total_words  = sum(len(d.page_content.split()) for d in docs)
        source_types = list({d.metadata.get("source_type", "unknown") for d in docs})

        return {
            "total_docs":     len(docs),
            "total_words":    total_words,
            "avg_words":      round(total_words / len(docs), 1),
            "source_types":   source_types,
            "has_ocr_pages":  any(d.metadata.get("ocr") for d in docs),
            "page_counts":    [d.metadata.get("page", i) for i, d in enumerate(docs)],
        }