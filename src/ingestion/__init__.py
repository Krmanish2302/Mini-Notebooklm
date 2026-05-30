"""
src/ingestion/__init__.py

Public surface for the ingestion package.
Import the compiled LangGraph app to run PDF (or any source) ingestion:

    from src.ingestion import ingestion_app, IngestionState
    result = ingestion_app.invoke({"file_path": "doc.pdf", "source_id": "doc1"})
"""
from .ingestion_graph import ingestion_app, IngestionState  # noqa: F401

__all__ = ["ingestion_app", "IngestionState"]
