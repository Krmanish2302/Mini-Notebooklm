"""
src/ingestion/__init__.py

Public API for the ingestion package.

Write path (typed — preferred, used by api.py):
    from src.ingestion.ingestion_router import ingest
    result = ingest(source_type="pdf", source_id="rep_01", file_path="doc.pdf")

Write path (untyped — fallback, used by master_pipeline.py):
    from src.ingestion import IngestionPipeline
    pipeline = IngestionPipeline()
    result = pipeline.ingest("doc.pdf", source_id="rep_01", source_type="pdf")
    # NOTE: source_type is required for the typed router;
    #       omitting it falls back to run_ingestion().

Read path:
    from src.ingestion import load_parent_retriever
    retriever = load_parent_retriever("data/vectorstores/rep_01")
"""

from .ingestion_runner  import run_ingestion          # noqa: F401
from .ingestion_graph   import ingestion_app          # noqa: F401
from .state             import IngestionState         # noqa: F401
from .parent_retriever  import load_parent_retriever  # noqa: F401


class IngestionPipeline:
    """
    Thin OOP wrapper so master_pipeline.py can call
        self._ingestion.ingest(source, source_id=..., source_type=...)

    If the caller supplies source_type=, the request is forwarded directly to
    ingestion_router.ingest() which dispatches to the correct specialised
    pipeline (pdf_pipeline, youtube_pipeline, text_pipeline, website_pipeline,
    image_pipeline).

    If source_type is NOT supplied, the call falls back to run_ingestion()
    which accepts only (file_path, source_id, source_type=None) — no extra
    kwargs are forwarded to avoid TypeError.

    NO auto-detection is performed here.  The caller (api.py form field,
    CLI arg, etc.) is always responsible for knowing the source type.
    """

    def ingest(
        self,
        source: str,
        *,
        source_id:   str | None = None,
        source_type: str | None = None,
        **kwargs,
    ) -> dict:
        """
        Ingest *source* into the vector store.

        Parameters
        ----------
        source       : File path, URL, or raw text string.
        source_id    : Unique ID for this source (auto-generated if omitted).
        source_type  : One of "pdf", "youtube", "website", "text", "image".
                       When supplied, routes to the matching specialised pipeline
                       via ingestion_router.  When omitted, falls back to
                       run_ingestion() (no chunk_size/chunk_overlap support).
        """
        import uuid
        sid = source_id or str(uuid.uuid4())[:8]

        if source_type:
            # Delegate entirely to the typed router — no detection needed
            from src.ingestion.ingestion_router import ingest as router_ingest
            import os
            is_file = False
            try:
                is_file = os.path.exists(source)
            except Exception:
                pass

            return router_ingest(
                source_type   = source_type,
                source_id     = sid,
                file_path     = source if (source_type != "text" or is_file) else None,
                content       = source if (source_type == "text" and not is_file) else None,
                strategy      = kwargs.get("strategy", "paragraph_based"),
                embedding_dim = kwargs.get("embedding_dim", 384),
            )

        # Fallback: run_ingestion only accepts (file_path, source_id, source_type)
        # Do NOT forward chunk_size/chunk_overlap — run_ingestion doesn't accept them
        return run_ingestion(
            source,
            source_id   = sid,
            source_type = kwargs.get("source_type"),
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
