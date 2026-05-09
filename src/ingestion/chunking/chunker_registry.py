from typing import Dict, Type
from .base_chunker import BaseChunker
from .recursive_chunker import RecursiveChunker
from .semantic_chunker import SemanticChunker
from .late_chunker import LateChunker
from .hierarchical_chunker import HierarchicalChunker
from .page_chunker import PageChunker
from .chapter_chunker import ChapterChunker
from .paragraph_chunker import ParagraphChunker

class ChunkerRegistry:
    """Registry for pluggable chunking strategies."""
    
    _chunkers: Dict[str, Type[BaseChunker]] = {
        "recursive": RecursiveChunker,
        "semantic": SemanticChunker,
        "late": LateChunker,
        "hierarchical": HierarchicalChunker,
        "page": PageChunker,
        "chapter": ChapterChunker,
        "paragraph": ParagraphChunker
    }
    
    @classmethod
    def get_chunker(cls, strategy: str, **kwargs) -> BaseChunker:
        if strategy not in cls._chunkers:
            raise ValueError(f"Unknown strategy: {strategy}. Available: {list(cls._chunkers.keys())}")
        return cls._chunkers[strategy](**kwargs)
    
    @classmethod
    def list_strategies(cls) -> list:
        return list(cls._chunkers.keys())