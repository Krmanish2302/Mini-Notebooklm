"""
contextual_enricher.py  —  NEW FILE

Pipeline position (from master_pipeline.py):
    CrossModalMerger.merge()  →  ContextualEnricher.enrich()  →  AdaptiveChunker

What it does:
    After cross-modal merging but BEFORE chunking, each raw text block is
    enriched with two types of contextual signal:

    1. Neighbour Context Window
       Each chunk gets a `context_window` field containing the N sentences
       before and after it from the original document.  This is NOT stored
       in the vector index — it is stored in SQLite metadata and surfaced
       at retrieval time so the LLM sees full surrounding context even when
       a small chunk matched.

    2. Source Header Injection
       A short header is prepended to every chunk:
           "Source: <title> (<source_type>)  |  Section: <section_heading>"
       This means even if a chunk is retrieved in isolation, the LLM always
       knows where it came from.

Usage:
    enricher = ContextualEnricher(window_sentences=3)
    enriched_chunks = enricher.enrich(chunks, metadata={"title": "...", "source_type": "pdf"})
"""
import re
from typing import List, Dict, Any, Optional


class ContextualEnricher:
    """
    Enriches chunks with surrounding context and source attribution.

    Args:
        window_sentences: Number of sentences to include before/after
                          each chunk as context.  Default 3.
        inject_header:    Whether to prepend a source header to each chunk.
                          Default True.
    """

    def __init__(
        self,
        window_sentences: int = 3,
        inject_header: bool = True,
    ):
        self.window_sentences = window_sentences
        self.inject_header = inject_header

    # ── public API ────────────────────────────────────────────────────────────

    def enrich(
        self,
        chunks: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Enrich a list of chunk dicts in-place.

        Args:
            chunks:   Output of AdaptiveChunker — list of chunk dicts.
            metadata: Source-level metadata dict (title, source_type, url, etc.)
                      from the original pipeline result.

        Returns:
            Same list with enriched fields added to each chunk:
            - chunk["context_window"]  : str  — surrounding sentences
            - chunk["content"]         : str  — original content (optionally
                                               with header prepended)
            - chunk["metadata"]["section_heading"] : str — nearest heading
        """
        metadata = metadata or {}
        if not chunks:
            return chunks

        # Build a sentence index over the full document for context windows
        full_text = " ".join(c["content"] for c in chunks)
        sentences = self._split_sentences(full_text)

        # Build a char-offset map: chunk_index → sentence_indices
        chunk_sentence_map = self._map_chunks_to_sentences(chunks, sentences)

        source_title = (
            metadata.get("title")
            or metadata.get("url", "")
            or metadata.get("name", "Unknown source")
        )
        source_type = metadata.get("source_type", "text")

        for i, chunk in enumerate(chunks):
            # 1. Neighbour context window
            sent_indices = chunk_sentence_map.get(i, [])
            chunk["context_window"] = self._build_context_window(
                sentences, sent_indices
            )

            # 2. Nearest section heading
            heading = self._find_nearest_heading(chunk["content"])
            chunk.setdefault("metadata", {})["section_heading"] = heading

            # 3. Source header injection
            if self.inject_header:
                header = self._build_header(
                    source_title=source_title,
                    source_type=source_type,
                    section=heading,
                )
                # Prepend header — keeps original content intact for retrieval
                chunk["content"] = f"{header}\n\n{chunk['content']}"

        return chunks

    # ── internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split text into sentences using simple regex (no NLTK dependency)."""
        # Split on . ! ? followed by whitespace or end-of-string
        raw = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in raw if s.strip()]

    def _map_chunks_to_sentences(
        self,
        chunks: List[Dict[str, Any]],
        sentences: List[str],
    ) -> Dict[int, List[int]]:
        """
        For each chunk, find which sentence indices it overlaps with.
        Simple linear scan — fast enough for typical document sizes.
        """
        mapping: Dict[int, List[int]] = {}
        sent_ptr = 0

        for chunk_idx, chunk in enumerate(chunks):
            chunk_words = set(chunk["content"].lower().split())
            matched = []
            for s_idx in range(sent_ptr, min(sent_ptr + 60, len(sentences))):
                sent_words = set(sentences[s_idx].lower().split())
                overlap = len(chunk_words & sent_words)
                if overlap >= max(1, len(sent_words) // 3):
                    matched.append(s_idx)
            if matched:
                sent_ptr = max(matched)  # advance pointer
            mapping[chunk_idx] = matched

        return mapping

    def _build_context_window(
        self,
        sentences: List[str],
        sent_indices: List[int],
    ) -> str:
        """Return N sentences before and after the matched sentence range."""
        if not sent_indices:
            return ""
        start = max(0, min(sent_indices) - self.window_sentences)
        end   = min(len(sentences), max(sent_indices) + self.window_sentences + 1)
        window = sentences[start:end]
        return " ".join(window)

    @staticmethod
    def _find_nearest_heading(content: str) -> str:
        """
        Try to extract a section heading from the chunk content.
        Looks for Markdown headings (## Heading) or ALL-CAPS lines.
        """
        # Markdown heading
        md_match = re.search(r"^#{1,3}\s+(.+)$", content, re.MULTILINE)
        if md_match:
            return md_match.group(1).strip()

        # ALL-CAPS short line (common in PDFs)
        lines = content.split("\n")
        for line in lines[:5]:  # check first 5 lines only
            stripped = line.strip()
            if stripped.isupper() and 3 < len(stripped) < 80:
                return stripped.title()

        return ""

    @staticmethod
    def _build_header(
        source_title: str,
        source_type: str,
        section: str,
    ) -> str:
        """
        Build a compact attribution header.
        e.g.:  "[Source: Annual Report 2024 (pdf) | Section: Financial Overview]"
        """
        parts = [f"Source: {source_title} ({source_type})"]
        if section:
            parts.append(f"Section: {section}")
        return f"[{' | '.join(parts)}]"
