"""
src/core — Shared config, domain models, and exceptions.

Usage:
    from src.core import settings, Source, Chunk, Query, LLMResponse
    from src.core import IngestionError, RetrievalError
    from src.core.config import get_settings
"""
from .config     import get_settings, Settings            # noqa: F401
from .models     import (                                  # noqa: F401
    Source, Chunk, Query, RetrievedChunk,
    Citation, LLMResponse, ChatMessage, Session,
)
from .exceptions import (                                  # noqa: F401
    MiniNotebookLMError,
    IngestionError,
    RetrievalError,
    GenerationError,
    StorageError,
    ConfigurationError,
    EvaluationError,
    GraphError,
    RateLimitError,
)

# Convenience singleton — same as get_settings() but importable directly
settings = get_settings()

__all__ = [
    # config
    "settings", "get_settings", "Settings",
    # models
    "Source", "Chunk", "Query", "RetrievedChunk",
    "Citation", "LLMResponse", "ChatMessage", "Session",
    # exceptions
    "MiniNotebookLMError", "IngestionError", "RetrievalError",
    "GenerationError", "StorageError", "ConfigurationError",
    "EvaluationError", "GraphError", "RateLimitError",
]