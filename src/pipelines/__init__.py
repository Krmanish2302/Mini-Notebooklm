"""
src/pipelines/__init__.py

Public API for all pipeline modes.

All three pipelines share the same RAG-based history contract:
  - Pass an `embedder` (any LangChain Embeddings with .embed_query())
  - Optionally pass `session_id` (default: "default")
  - History is stored in SQLite and retrieved semantically per turn.
  - No ConversationBufferWindowMemory or MemorySaver is used.

Typical setup::

    from src.pipelines import ChatPipeline, StudyPipeline, DeepResearchPipeline
    from src.ingestion.embedding.embedding_registry import EmbeddingRegistry

    embedder = EmbeddingRegistry.get()   # your existing model

    chat   = ChatPipeline(faiss, sqlite, sources, embedder, session_id="user-1")
    study  = StudyPipeline(faiss, sqlite, sources, embedder, session_id="user-1")
    deep   = DeepResearchPipeline(faiss, sqlite, sources, embedder, session_id="user-1")

    chat.set_llm(LLMRegistry.get())
    result = chat.run("What is photosynthesis?")
"""
from .chat_pipeline          import ChatPipeline           # noqa: F401
from .study_pipeline         import StudyPipeline          # noqa: F401
from .deep_research_pipeline import DeepResearchPipeline   # noqa: F401

__all__ = ["ChatPipeline", "StudyPipeline", "DeepResearchPipeline"]
