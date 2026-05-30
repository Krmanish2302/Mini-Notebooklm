"""
exceptions.py — Structured exception hierarchy for Mini NotebookLM.

Every exception carries:
  - message   : human-readable description
  - context   : optional dict of diagnostic key-values (logged, never shown to users)
  - http_status: maps to FastAPI HTTP responses

LangChain integration:
  - LangChainError wraps any langchain_core exception so it travels through
    the pipeline's error handling nodes and surfaces cleanly to the API layer.
  - from_lc_error() is the canonical way to wrap a LangChain exception.

Usage:
    raise IngestionError("PDF parse failed", context={"file": path, "page": 3})
    raise RateLimitError("Groq rate limit hit", context={"provider": "groq", "retry_after": 30})

    # In FastAPI exception handlers:
    @app.exception_handler(MiniNotebookLMError)
    async def handle(request, exc):
        return JSONResponse(
            status_code=exc.http_status,
            content={"error": exc.error_code, "message": str(exc)},
        )
"""
from __future__ import annotations
from typing import Any, Dict, Optional


class MiniNotebookLMError(Exception):
    """
    Base exception for all Mini NotebookLM errors.

    Attributes
    ----------
    message     : str  — what went wrong
    context     : dict — diagnostic data (file paths, chunk IDs, etc.)
    http_status : int  — maps to HTTP response code
    error_code  : str  — machine-readable snake_case identifier
    """

    http_status: int  = 500
    error_code:  str  = "internal_error"

    def __init__(
        self,
        message: str = "An unexpected error occurred",
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message  = message
        self.context  = context or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error":      self.error_code,
            "message":    self.message,
            "http_status": self.http_status,
            "context":    self.context,
        }

    def __repr__(self) -> str:
        ctx = f" | context={self.context}" if self.context else ""
        return f"{self.__class__.__name__}({self.message!r}{ctx})"


# ── Ingestion ──────────────────────────────────────────────────────────────────

class IngestionError(MiniNotebookLMError):
    """File parsing or chunking failed."""
    http_status = 422
    error_code  = "ingestion_error"


class UnsupportedFileTypeError(IngestionError):
    """File extension not in allowed list."""
    http_status = 415
    error_code  = "unsupported_file_type"


class FileTooLargeError(IngestionError):
    """File exceeds max_file_size_mb limit."""
    http_status = 413
    error_code  = "file_too_large"


# ── Retrieval ─────────────────────────────────────────────────────────────────

class RetrievalError(MiniNotebookLMError):
    """Vector search, BM25, or graph traversal failed."""
    http_status = 500
    error_code  = "retrieval_error"


class NoResultsError(RetrievalError):
    """Query returned zero results — not a system error."""
    http_status = 404
    error_code  = "no_results"


# ── Generation ────────────────────────────────────────────────────────────────

class GenerationError(MiniNotebookLMError):
    """LLM invocation failed."""
    http_status = 502
    error_code  = "generation_error"


class RateLimitError(GenerationError):
    """Provider rate limit or quota exceeded."""
    http_status = 429
    error_code  = "rate_limit_error"


class PromptTooLongError(GenerationError):
    """Assembled prompt exceeds the model's context window."""
    http_status = 422
    error_code  = "prompt_too_long"


# ── Storage ───────────────────────────────────────────────────────────────────

class StorageError(MiniNotebookLMError):
    """SQLite, FAISS, or file I/O operation failed."""
    http_status = 500
    error_code  = "storage_error"


class SourceNotFoundError(StorageError):
    """Requested source_id does not exist."""
    http_status = 404
    error_code  = "source_not_found"


# ── Graph ─────────────────────────────────────────────────────────────────────

class GraphError(MiniNotebookLMError):
    """Knowledge graph operation failed."""
    http_status = 500
    error_code  = "graph_error"


# ── Evaluation ────────────────────────────────────────────────────────────────

class EvaluationError(MiniNotebookLMError):
    """RAGAS evaluation pipeline failed."""
    http_status = 500
    error_code  = "evaluation_error"


# ── Configuration ─────────────────────────────────────────────────────────────

class ConfigurationError(MiniNotebookLMError):
    """Invalid or missing configuration value."""
    http_status = 500
    error_code  = "configuration_error"


# ── LangChain bridge ──────────────────────────────────────────────────────────

class LangChainError(MiniNotebookLMError):
    """
    Wraps any exception originating from a LangChain component so it
    travels cleanly through the pipeline's error handling nodes.

    Usage:
        try:
            result = llm.invoke(prompt)
        except Exception as exc:
            raise LangChainError.from_lc_error(exc, component="ChatGroq")
    """
    http_status = 502
    error_code  = "langchain_error"

    @classmethod
    def from_lc_error(
        cls,
        exc:       Exception,
        component: str = "LangChain",
    ) -> "LangChainError":
        return cls(
            message=f"{component} error: {exc}",
            context={"component": component, "original": str(exc), "type": type(exc).__name__},
        )