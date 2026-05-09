"""
content_analyzer.py

Analyzes raw text content to recommend the best chunking strategy.

Fix: analyze() now accepts (content, source_type) OR just (content,) safely.
     source_type defaults to "text" so callers that omit it don't crash.
"""
import re
from typing import Dict, Any, Optional


class ContentAnalyzer:
    """Analyzes content structure and recommends a chunking strategy."""

    def __init__(
        self,
        sample_paragraphs: int = 15,
        embedding_model_max_tokens: int = 384,
    ):
        self.sample_paragraphs = sample_paragraphs
        self.embedding_model_max_tokens = embedding_model_max_tokens

    # ── public API ────────────────────────────────────────────────────────────

    def analyze(
        self,
        content: str,
        source_type: Optional[str] = "text",   # FIX: was required, now optional
    ) -> Dict[str, Any]:
        """
        Analyze *content* and return strategy recommendation.

        Args:
            content:     Plain-text content to inspect.
            source_type: One of pdf | website | youtube | csv | image | text.
                         Defaults to "text" if not provided.

        Returns:
            {
                sampled_length, estimated_tokens, avg_tokens_per_paragraph,
                paragraph_count, word_count,
                structure: {has_chapters, has_page_markers, has_headings},
                recommendation: {strategy, reason, chunk_size}
            }
        """
        source_type = (source_type or "text").lower().strip()
        sample = self._get_sample(content)

        words = sample.split()
        estimated_tokens = int(len(words) / 0.75)  # ~0.75 words per token
        paragraphs = [p for p in sample.split("\n\n") if p.strip()]
        avg_tokens_per_paragraph = estimated_tokens // max(len(paragraphs), 1)

        # Structural signals
        has_chapters = bool(
            re.search(r"^(Chapter|CHAPTER|Ch\.)\s+\d+", sample, re.MULTILINE)
        )
        has_page_markers = "[Page" in content
        has_headings = bool(re.search(r"^#{1,3}\s", sample, re.MULTILINE))
        has_timestamps = bool(
            re.search(r"\[\d{2}:\d{2}\]", content)  # YouTube [MM:SS] markers
        )

        recommendation = self._recommend_strategy(
            source_type=source_type,
            has_chapters=has_chapters,
            has_page_markers=has_page_markers,
            has_headings=has_headings,
            has_timestamps=has_timestamps,
            avg_tokens=avg_tokens_per_paragraph,
            word_count=len(words),
        )

        return {
            "sampled_length": len(sample),
            "estimated_tokens": estimated_tokens,
            "avg_tokens_per_paragraph": avg_tokens_per_paragraph,
            "paragraph_count": len(paragraphs),
            "word_count": len(content.split()),
            "structure": {
                "has_chapters": has_chapters,
                "has_page_markers": has_page_markers,
                "has_headings": has_headings,
                "has_timestamps": has_timestamps,
            },
            "recommendation": recommendation,
        }

    # ── internals ─────────────────────────────────────────────────────────────

    def _get_sample(self, content: str) -> str:
        paragraphs = content.split("\n\n")
        if len(paragraphs) <= self.sample_paragraphs:
            return content
        return "\n\n".join(paragraphs[: self.sample_paragraphs])

    def _recommend_strategy(
        self,
        source_type: str,
        has_chapters: bool,
        has_page_markers: bool,
        has_headings: bool,
        has_timestamps: bool,
        avg_tokens: int,
        word_count: int,
    ) -> Dict[str, Any]:
        t = self.embedding_model_max_tokens

        if source_type in ("youtube", "transcript") or has_timestamps:
            return {
                "strategy": "paragraph",
                "reason": "Transcript content — paragraph boundaries map to topic shifts",
                "chunk_size": f"~{t} tokens",
            }
        if source_type == "website":
            return {
                "strategy": "recursive",
                "reason": "Web content benefits from recursive splitting",
                "chunk_size": f"{t} tokens",
            }
        if has_chapters:
            return {
                "strategy": "chapter",
                "reason": "Document has clear chapter headings",
                "chunk_size": "1 chapter per chunk",
            }
        if has_page_markers and avg_tokens > 200:
            return {
                "strategy": "page",
                "reason": "Dense page-structured document",
                "chunk_size": "1 page per chunk",
            }
        if has_headings or word_count > 5000:
            return {
                "strategy": "hierarchical",
                "reason": "Long / heading-structured document — hierarchical preserves context",
                "chunk_size": f"{t} tokens (parent) / {t // 2} tokens (child)",
            }
        return {
            "strategy": "recursive",
            "reason": "Default safe strategy for general text",
            "chunk_size": f"{t} tokens with overlap",
        }
