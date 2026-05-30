"""
cross_modal_merger.py

Merges Documents from multiple source types into a single ordered list.
Deduplicates by content hash. Sorts by source_type priority.
No LLM calls.
"""
from __future__ import annotations
import hashlib
from typing import List
from langchain_core.documents import Document

_PRIORITY = {"pdf": 0, "website": 1, "youtube": 2, "csv": 3, "text": 4}


class CrossModalMerger:
    def merge(self, *doc_lists: List[Document]) -> List[Document]:
        """
        Merge multiple Document lists, deduplicate, sort by source priority.

        Args:
            *doc_lists: Any number of List[Document] to merge.

        Returns:
            Single deduplicated, sorted List[Document].
        """
        seen:   set            = set()
        merged: List[Document] = []

        for docs in doc_lists:
            for doc in docs:
                h = hashlib.sha256(doc.page_content.encode()).hexdigest()
                if h not in seen:
                    seen.add(h)
                    merged.append(doc)

        merged.sort(key=lambda d: _PRIORITY.get(d.metadata.get("source_type", "text"), 99))
        return merged