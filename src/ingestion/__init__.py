"""
src/ingestion/__init__.py

Public surface for the ingestion package.

Ingestion (write path):
    from src.ingestion import ingestion_app, IngestionState
    result = ingestion_app.invoke({"file_path": "doc.pdf", "source_id": "doc1"})

Retrieval (read path) — use after ingestion:
    from src.ingestion import load_parent_retriever
    retriever = load_parent_retriever("data/vectorstores/doc1")
    docs = retriever.invoke("your query here")
"""
from .ingestion_graph    import ingestion_app, IngestionState  # noqa: F401
from .parent_retriever   import load_parent_retriever           # noqa: F401

__all__ = ["ingestion_app", "IngestionState", "load_parent_retriever"]
