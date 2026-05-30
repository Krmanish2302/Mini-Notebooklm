"""
models.py — Core Pydantic v2 domain models.

LangChain integration
---------------------
  Chunk.to_document()      → langchain_core.documents.Document
  LLMResponse.to_generation() → langchain_core.outputs.ChatGeneration
  RetrievedChunk.to_document() → Document with score in metadata

These bridges let pipeline code pass domain models directly into
LangChain chains, output parsers, and callback handlers.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ── Source ────────────────────────────────────────────────────────────────────

class Source(BaseModel):
    """Represents an uploaded or linked source document."""

    id:          str
    title:       str
    source_type: Literal["pdf", "image", "video", "audio", "website", "youtube", "csv", "text"]
    file_path:   Optional[str]            = None
    url:         Optional[str]            = None
    metadata:    Dict[str, Any]           = Field(default_factory=dict)
    created_at:  datetime                 = Field(default_factory=_utcnow)
    status:      Literal["processing", "ready", "error"] = "processing"

    @field_validator("title")
    @classmethod
    def title_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("title must not be empty")
        return v.strip()

    @model_validator(mode="after")
    def require_path_or_url(self) -> "Source":
        if self.file_path is None and self.url is None:
            raise ValueError("Source must have either file_path or url")
        return self


# ── Chunk ─────────────────────────────────────────────────────────────────────

class Chunk(BaseModel):
    """
    Text chunk with optional embedding.

    .to_document() converts to a LangChain Document so chunks can flow
    directly into retrievers, prompt templates, and output parsers.
    """

    id:             str
    source_id:      str
    content:        str
    modality:       Literal["text", "image_caption", "transcript", "table"] = "text"
    embedding:      Optional[List[float]]  = None
    metadata:       Dict[str, Any]         = Field(default_factory=dict)
    related_chunks: List[str]              = Field(default_factory=list)
    page_number:    Optional[int]          = None
    timestamp:      Optional[str]          = None

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Chunk content must not be empty")
        return v

    def to_document(self):
        """
        Convert to langchain_core.documents.Document.
        All Chunk fields land in metadata so retrievers can access them.
        """
        from langchain_core.documents import Document
        return Document(
            page_content=self.content,
            metadata={
                "chunk_id":      self.id,
                "source_id":     self.source_id,
                "modality":      self.modality,
                "page_number":   self.page_number,
                "timestamp":     self.timestamp,
                "related_chunks": self.related_chunks,
                **self.metadata,
            },
        )

    @classmethod
    def from_document(cls, doc, source_id: str = "", chunk_id: str = "") -> "Chunk":
        """Reconstruct a Chunk from a LangChain Document."""
        meta = doc.metadata or {}
        return cls(
            id=chunk_id or meta.get("chunk_id", ""),
            source_id=source_id or meta.get("source_id", ""),
            content=doc.page_content,
            modality=meta.get("modality", "text"),
            page_number=meta.get("page_number"),
            timestamp=meta.get("timestamp"),
            related_chunks=meta.get("related_chunks", []),
            metadata={k: v for k, v in meta.items()
                      if k not in {"chunk_id","source_id","modality",
                                   "page_number","timestamp","related_chunks"}},
        )


# ── Query ─────────────────────────────────────────────────────────────────────

class Query(BaseModel):
    """User query with mode and session context."""

    text:            str
    mode:            Literal["chat", "deep_research", "study"] = "chat"
    session_id:      str
    source_ids:      Optional[List[str]] = None
    history_context: Optional[str]       = None

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Query text must not be empty")
        if len(v) > 4000:
            raise ValueError("Query text must be ≤ 4000 characters")
        return v.strip()


# ── RetrievedChunk ────────────────────────────────────────────────────────────

class RetrievedChunk(BaseModel):
    """Chunk + retrieval metadata returned by any retriever."""

    chunk:            Chunk
    score:            float = Field(ge=0.0, le=1.0)
    retrieval_method: Literal["dense", "sparse", "graph", "hybrid"]

    def to_document(self):
        """
        Convert to LangChain Document with retrieval score in metadata.
        Consistent with LangChain's (Document, score) tuple convention.
        """
        doc = self.chunk.to_document()
        doc.metadata["score"]            = round(self.score, 4)
        doc.metadata["retrieval_method"] = self.retrieval_method
        return doc

    @classmethod
    def from_document(cls, doc, score: float = 0.0,
                      method: str = "hybrid") -> "RetrievedChunk":
        """Reconstruct from a LangChain Document."""
        chunk = Chunk.from_document(doc)
        return cls(chunk=chunk, score=score, retrieval_method=method)


# ── Citation ──────────────────────────────────────────────────────────────────

class Citation(BaseModel):
    """Inline citation linking an answer sentence to its source chunk."""

    label:      str            = ""    # e.g. "S1", "S2"
    text:       str
    source_id:  str
    chunk_id:   str
    confidence: float = Field(ge=0.0, le=1.0)


# ── LLMResponse ───────────────────────────────────────────────────────────────

class LLMResponse(BaseModel):
    """Structured output from the generation pipeline."""

    content:     str
    citations:   List[Citation] = Field(default_factory=list)
    confidence:  float          = Field(ge=0.0, le=1.0)
    mode:        str
    tokens_used: int            = 0
    latency_ms:  float          = 0.0
    follow_ups:  List[str]      = Field(default_factory=list)
    ragas:       Optional[Dict[str, Any]] = None

    def to_generation(self):
        """
        Convert to langchain_core.outputs.ChatGeneration.
        Useful when LLMResponse needs to pass through a LangChain output parser.
        """
        from langchain_core.outputs import ChatGeneration
        from langchain_core.messages import AIMessage
        return ChatGeneration(
            text=self.content,
            message=AIMessage(content=self.content),
            generation_info={
                "citations":   [c.model_dump() for c in self.citations],
                "tokens_used": self.tokens_used,
                "latency_ms":  self.latency_ms,
                "mode":        self.mode,
            },
        )


# ── ChatMessage ───────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    """Single chat turn stored in the session."""

    id:           str
    session_id:   str
    role:         Literal["user", "assistant", "system"]
    content:      str
    timestamp:    datetime      = Field(default_factory=_utcnow)
    sources_used: List[str]     = Field(default_factory=list)

    def to_lc_message(self):
        """
        Convert to a LangChain BaseMessage subclass.
        Used to hydrate ConversationBufferWindowMemory from stored history.
        """
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
        if self.role == "user":
            return HumanMessage(content=self.content)
        if self.role == "assistant":
            return AIMessage(content=self.content)
        return SystemMessage(content=self.content)


# ── Session ───────────────────────────────────────────────────────────────────

class Session(BaseModel):
    """Chat session container."""

    id:             str
    mode:           Literal["chat", "deep_research", "study"]
    created_at:     datetime        = Field(default_factory=_utcnow)
    messages:       List[ChatMessage] = Field(default_factory=list)
    active_sources: List[str]         = Field(default_factory=list)

    def to_lc_messages(self) -> list:
        """
        Return all messages as a List[BaseMessage] for LangChain memory hydration.
        Usage:
            memory = ConversationBufferWindowMemory(k=8)
            memory.chat_memory.messages = session.to_lc_messages()
        """
        return [m.to_lc_message() for m in self.messages]

    def last_n_messages(self, n: int = 8) -> List[ChatMessage]:
        return self.messages[-n:]