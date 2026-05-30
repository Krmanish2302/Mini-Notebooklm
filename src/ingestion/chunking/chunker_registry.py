"""
chunker_registry.py

Central registry mapping string names → chunker instances.
Used by master_pipeline.py and plugin_config.yaml.
"""
from __future__ import annotations
from typing import Dict
from .base_chunker       import BaseChunker
from .recursive_chunker  import RecursiveChunker
from .semantic_chunker   import SemanticChunker
from .sentence_chunker   import SentenceChunker
from .hierarchical_chunker import HierarchicalChunker
from .adaptive_chunker   import AdaptiveChunker
from .page_chunker       import PageChunker
from .paragraph_chunker  import ParagraphChunker
from .chapter_chunker    import ChapterChunker


class ChunkerRegistry:
    _registry: Dict[str, BaseChunker] = {
        "recursive":    RecursiveChunker(),
        "semantic":     SemanticChunker(),
        "sentence":     SentenceChunker(),
        "hierarchical": HierarchicalChunker(),
        "adaptive":     AdaptiveChunker(),
        "page":         PageChunker(),
        "paragraph":    ParagraphChunker(),
        "chapter":      ChapterChunker(),
    }

    @classmethod
    def get(cls, name: str) -> BaseChunker:
        chunker = cls._registry.get(name)
        if not chunker:
            raise ValueError(f"Unknown chunker '{name}'. Available: {list(cls._registry)}")
        return chunker

    @classmethod
    def available(cls) -> list:
        return list(cls._registry.keys())