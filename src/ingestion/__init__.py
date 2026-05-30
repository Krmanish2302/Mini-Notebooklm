"""
src/ingestion/__init__.py

Public API for the ingestion package.

Write path (ingest a file):
    from src.ingestion import run_ingestion
    result = run_ingestion("data/report.pdf", source_id="report_001")

Read path (load retriever after ingestion):
    from src.ingestion import load_parent_retriever
    retriever = load_parent_retriever("data/vectorstores/report_001")
    docs = retriever.invoke("What is the conclusion?")

Pipeline class (used by master_pipeline.py):
    from src.ingestion import IngestionPipeline
    pipeline = IngestionPipeline()
    result = pipeline.ingest("doc.pdf", source_id="doc_001")
"""

from .ingestion_runner import run_ingestion  # noqa: F401
from .ingestion_graph import ingestion_app  # noqa: F401
from .state import IngestionState  # noqa: F401
from .parent_retriever import load_parent_retriever  # noqa: F401


class IngestionPipeline:
    """
    Thin wrapper around run_ingestion() so that master_pipeline.py can call
        self._ingestion.ingest(source, source_id=..., ...)
    using the same object-oriented interface regardless of what the runner
    module exposes.
    """

    def ingest(
        self,
        source: str,
        *,
        source_id: str | None = None,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        **kwargs,
    ) -> dict:
        """
        Ingest *source* (file path, URL, or raw text).

        Delegates to src.ingestion.ingestion_router.ingest() when the
        source_type can be inferred, otherwise falls back to run_ingestion().
        """
        import os
        import uuid

        sid = source_id or str(uuid.uuid4())[:8]

        # Try the typed router first (supports PDF, YouTube, website, text, image)
        try:
            from src.ingestion.file_detector import detect_source_type
            from src.ingestion.ingestion_router import ingest as router_ingest

            source_type = detect_source_type(source)
            return router_ingest(
                source_type=source_type,
                source_id=sid,
                file_path=source if source_type != "text" or os.path.isfile(source) else None,
                content=source if source_type == "text" and not os.path.isfile(source) else None,
                strategy=kwargs.get("strategy", "paragraph_based"),
                embedding_dim=kwargs.get("embedding_dim", 384),
            )
        except Exception:
            pass

        # Fallback: legacy run_ingestion
        return run_ingestion(
            source,
            source_id=sid,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            **kwargs,
        )

    def list_sources(self) -> list:
        """Return a list of ingested source metadata dicts (best-effort)."""
        try:
            from src.ingestion.ingestion_runner import list_sources as _ls
            return _ls()
        except Exception:
            return []

    def delete_source(self, source_id: str) -> bool:
        """Delete a source by ID (best-effort)."""
        try:
            from src.ingestion.ingestion_runner import delete_source as _ds
            return _ds(source_id)
        except Exception:
            return False


__all__ = [
    "run_ingestion",
    "ingestion_app",
    "IngestionState",
    "load_parent_retriever",
    "IngestionPipeline",
]
