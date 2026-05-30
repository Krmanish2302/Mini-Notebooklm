"""
src/pipelines/__init__.py

Public API for all pipeline modes.

All three pipelines share the same RAG-based history contract:
  - Pass an `embedder` (any LangChain Embeddings with .embed_query())
  - Optionally pass `session_id` (default: "default")
  - History is stored in SQLite and retrieved semantically per turn.
  - No ConversationBufferWindowMemory or MemorySaver is used.

Typical setup (manual)::

    from src.pipelines import ChatPipeline, StudyPipeline, DeepResearchPipeline
    from src.ingestion.embedding.embedding_registry import EmbeddingRegistry

    embedder = EmbeddingRegistry.get_default()
    chat   = ChatPipeline(faiss, sqlite, sources, embedder, session_id="user-1")
    result = chat.run("What is photosynthesis?", source_ids=["bio_101"])

Typical setup (Streamlit — preferred)::

    from src.pipelines import PipelineFactory

    factory = PipelineFactory.from_session(st.session_state, session_id="user-1")
    result  = factory.get("chat").run("What is photosynthesis?", source_ids=["bio_101"])

Ingest::

    from src.pipelines import run_ingest
    state = run_ingest(source_type="pdf", source_id="doc1", source_input="paper.pdf")
"""
from .chat_pipeline          import ChatPipeline           # noqa: F401
from .study_pipeline         import StudyPipeline          # noqa: F401
from .deep_research_pipeline import DeepResearchPipeline   # noqa: F401
from .pipeline_factory       import PipelineFactory        # noqa: F401
from .ingest_graph           import run_ingest             # noqa: F401

__all__ = [
    "ChatPipeline",
    "StudyPipeline",
    "DeepResearchPipeline",
    "PipelineFactory",
    "run_ingest",
]
