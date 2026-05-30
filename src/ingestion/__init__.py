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
"""

from .ingestion_runner import run_ingestion  # noqa: F401
from .ingestion_graph import ingestion_app  # noqa: F401
from .state import IngestionState  # noqa: F401
from .parent_retriever import load_parent_retriever  # noqa: F401

__all__ = [
    "run_ingestion",
    "ingestion_app",
    "IngestionState",
    "load_parent_retriever",
]
