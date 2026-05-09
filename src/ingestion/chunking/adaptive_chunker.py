"""
adaptive_chunker.py

Single entry-point for all chunking strategies.
Internally delegates to ChunkerRegistry — callers only need this class.

Usage (in master_pipeline or api.py):
    chunker = AdaptiveChunker()
    chunks  = chunker.chunk(content, strategy="recursive", metadata={...})
    # or: previews = chunker.chunk(file_path_or_text, strategy="semantic", ...)
"""
from typing import List, Dict, Any, Optional
from .chunker_registry import ChunkerRegistry


class AdaptiveChunker:
    """
    Adaptive chunker that selects the right strategy at call-time.

    Available strategies (from ChunkerRegistry):
        recursive   – LangChain RecursiveCharacterTextSplitter  (default, fast)
        semantic    – Embedding-based boundary detection (slower, higher quality)
        hierarchical – Parent / child chunk pairs
        paragraph   – Split on blank lines
        page        – Split on page markers  ([Page N])
        chapter     – Split on heading patterns
        late        – Late chunking for long-context models
    """

    DEFAULT_STRATEGY = "recursive"

    def __init__(self, default_strategy: str = DEFAULT_STRATEGY):
        self.default_strategy = default_strategy

    # ── public API ────────────────────────────────────────────────────────────

    def chunk(
        self,
        content: str,
        strategy: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Chunk *content* using the requested strategy.

        Args:
            content:  Plain-text content to split.
            strategy: Chunking strategy name.  Defaults to self.default_strategy.
            metadata: Dict attached to every produced chunk
                      (must include 'source_id').
            **kwargs: Forwarded to the underlying chunker constructor
                      (e.g. chunk_size=512, chunk_overlap=64).

        Returns:
            List of chunk dicts:
            [
                {
                    "id":       "<source_id>_chunk_<i>",
                    "content":  "<text>",
                    "metadata": { ...metadata, "chunk_index": i,
                                  "strategy": "<strategy>" },
                    "modality": "text" | "transcript" | ...
                },
                ...
            ]
        """
        strategy = (strategy or self.default_strategy).lower().strip()
        metadata = metadata or {}

        # Build the chunker — kwargs let callers override chunk_size etc.
        chunker = ChunkerRegistry.get_chunker(strategy, **kwargs)
        chunks  = chunker.chunk(content, metadata=metadata)

        # Stamp every chunk with the strategy that produced it
        for chunk in chunks:
            chunk.setdefault("metadata", {})["strategy"] = strategy

        return chunks

    def list_strategies(self) -> List[str]:
        """Return all registered strategy names."""
        return ChunkerRegistry.list_strategies()

    def recommend_strategy(
        self,
        word_count: int,
        modality: str = "text",
        has_pages: bool = False,
        has_chapters: bool = False,
    ) -> str:
        """
        Lightweight heuristic to pick a strategy without running an analyzer.

        Rules (in priority order):
            - Transcript (YouTube / audio)  →  paragraph
            - Has chapter headings          →  chapter
            - Has page markers              →  page
            - Long document (> 5000 words)  →  hierarchical
            - Short / medium text           →  recursive
        """
        if modality == "transcript":
            return "paragraph"
        if has_chapters:
            return "chapter"
        if has_pages:
            return "page"
        if word_count > 5000:
            return "hierarchical"
        return "recursive"
