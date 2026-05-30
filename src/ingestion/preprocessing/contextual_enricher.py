"""
contextual_enricher.py

Enriches Documents with:
  1. Surrounding sentence context window (stored in metadata)
  2. Source attribution header prepended to page_content
  3. Nearest section heading detection

No LLM calls — pure regex and string operations.
"""
from __future__ import annotations
import re
from typing import Any, Dict, List, Optional
from langchain_core.documents import Document


class ContextualEnricher:
    def __init__(self, window_sentences: int = 3, inject_header: bool = True):
        self.window_sentences = window_sentences
        self.inject_header    = inject_header

    def enrich(self, docs: List[Document], metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        metadata     = metadata or {}
        full_text    = " ".join(d.page_content for d in docs)
        sentences    = self._split_sentences(full_text)
        source_title = metadata.get("title") or metadata.get("url", "Unknown source")
        source_type  = metadata.get("source_type", "text")
        result       = []

        for i, doc in enumerate(docs):
            content = doc.page_content
            heading = self._find_heading(content)
            context = self._context_window(sentences, content)

            if self.inject_header:
                header  = f"[Source: {source_title} ({source_type}){' | Section: ' + heading if heading else ''}]"
                content = f"{header}\n\n{content}"

            result.append(Document(
                page_content=content,
                metadata={
                    **doc.metadata,
                    "context_window":  context,
                    "section_heading": heading,
                },
            ))
        return result

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]

    def _context_window(self, sentences: List[str], content: str) -> str:
        words = set(content.lower().split())
        for i, s in enumerate(sentences):
            if len(words & set(s.lower().split())) > 3:
                start = max(0, i - self.window_sentences)
                end   = min(len(sentences), i + self.window_sentences + 1)
                return " ".join(sentences[start:end])
        return ""

    @staticmethod
    def _find_heading(content: str) -> str:
        m = re.search(r"^#{1,3}\s+(.+)$", content, re.MULTILINE)
        if m:
            return m.group(1).strip()
        for line in content.split("\n")[:5]:
            s = line.strip()
            if s.isupper() and 3 < len(s) < 80:
                return s.title()
        return ""