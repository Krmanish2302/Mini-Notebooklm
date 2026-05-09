from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Literal
from datetime import datetime
import numpy as np

class Source(BaseModel):
    """Represents an uploaded or searched source."""
    id: str
    title: str
    source_type: Literal["pdf", "image", "video", "audio", "website", "youtube", "csv", "text"]
    file_path: Optional[str] = None
    url: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    status: Literal["processing", "ready", "error"] = "processing"

class Chunk(BaseModel):
    """Represents a text chunk with embedding."""
    id: str
    source_id: str
    content: str
    modality: Literal["text", "image_caption", "transcript", "table"]
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    related_chunks: List[str] = Field(default_factory=list)
    page_number: Optional[int] = None
    timestamp: Optional[str] = None

class Query(BaseModel):
    """User query with mode and context."""
    text: str
    mode: Literal["chat", "deep_research", "study"] = "chat"
    session_id: str
    source_ids: Optional[List[str]] = None
    history_context: Optional[str] = None

class RetrievedChunk(BaseModel):
    """Chunk retrieved from vector store."""
    chunk: Chunk
    score: float
    retrieval_method: Literal["dense", "sparse", "graph", "hybrid"]

class Citation(BaseModel):
    """Citation with source reference."""
    text: str
    source_id: str
    chunk_id: str
    confidence: float = Field(ge=0.0, le=1.0)

class LLMResponse(BaseModel):
    """Structured LLM response."""
    content: str
    citations: List[Citation] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    mode: str
    tokens_used: int
    latency_ms: float

class ChatMessage(BaseModel):
    """Single chat message."""
    id: str
    session_id: str
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    sources_used: List[str] = Field(default_factory=list)

class Session(BaseModel):
    """Chat session."""
    id: str
    mode: Literal["chat", "deep_research", "study"]
    created_at: datetime = Field(default_factory=datetime.now)
    messages: List[ChatMessage] = Field(default_factory=list)
    active_sources: List[str] = Field(default_factory=list)