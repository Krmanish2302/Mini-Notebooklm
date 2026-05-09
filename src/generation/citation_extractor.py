"""CitationExtractor — maps [SOURCE_X] markers to real chunk metadata.

Phase 3 deliverable: correct interface + full wire-up.
Phase 4 will extend inject() to embed clickable references in the UI.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


class CitationExtractor:
    """Resolves [SOURCE_X] markers in LLM responses to real chunk metadata.

    Usage
    -----
    extractor = CitationExtractor()

    # After ResponseGenerator.generate():
    annotated = extractor.inject(response=text, documents=docs)

    # Or extract citation metadata for API consumers:
    cites = extractor.extract(response=text, documents=docs)
    """

    _CITATION_RE = re.compile(r'\[SOURCE_(\d+)\]')

    def extract(
        self,
        response: str,
        documents: List[Any],
    ) -> List[Dict[str, Any]]:
        """Return a list of resolved citation dicts.

        Each dict has:
            source_index : int  — 1-based index found in [SOURCE_X]
            source_id    : str  — from chunk metadata (if available)
            source_name  : str  — human-readable name / URL
            chunk_excerpt: str  — first 200 chars of the cited chunk
            confidence   : float — 1.0 (direct citation marker)
        """
        raw_indices = self._CITATION_RE.findall(response)
        seen = set()
        citations: List[Dict[str, Any]] = []

        for idx_str in raw_indices:
            idx = int(idx_str)          # 1-based
            if idx in seen:
                continue
            seen.add(idx)

            doc_idx = idx - 1           # 0-based list index
            if 0 <= doc_idx < len(documents):
                doc = documents[doc_idx]
                if hasattr(doc, "page_content"):
                    content = doc.page_content
                    meta = doc.metadata or {}
                else:
                    content = doc.get("content", "")
                    meta = {k: v for k, v in doc.items() if k != "content"}

                citations.append({
                    "source_index": idx,
                    "source_id":    meta.get("source_id", meta.get("id", f"source_{idx}")),
                    "source_name":  meta.get("source", meta.get("url", meta.get("file_path", ""))),
                    "chunk_excerpt": content[:200],
                    "confidence":   1.0,
                })
            else:
                # Marker points to a document that wasn't retrieved — record it
                # so callers can detect hallucinated citations.
                citations.append({
                    "source_index": idx,
                    "source_id":    f"source_{idx}",
                    "source_name":  "",
                    "chunk_excerpt": "",
                    "confidence":   0.0,  # unresolvable → low confidence
                })

        return citations

    def inject(
        self,
        response: str,
        documents: List[Any],
        fmt: str = "inline",
    ) -> str:
        """Inject resolved citation labels into the response string.

        fmt='inline'  : replace [SOURCE_X] with [X: source_name] or [X]
        fmt='preserve': return response unchanged (citations kept as-is)
        """
        if fmt == "preserve" or not documents:
            return response

        citations = {c["source_index"]: c for c in self.extract(response, documents)}

        def _replace(match: re.Match) -> str:
            idx = int(match.group(1))
            cite = citations.get(idx)
            if cite and cite["source_name"]:
                short = cite["source_name"].split("/")[-1][:40]  # last path segment
                return f"[{idx}: {short}]"
            return f"[{idx}]"

        return self._CITATION_RE.sub(_replace, response)

    def extract_unresolvable(
        self, response: str, documents: List[Any]
    ) -> List[int]:
        """Return 1-based indices of [SOURCE_X] markers that have no matching
        document — i.e., potential hallucinated citations."""
        return [
            c["source_index"]
            for c in self.extract(response, documents)
            if c["confidence"] < 1.0
        ]
