"""
cross_modal_merger.py  —  TOMBSTONED

This module has been intentionally removed.

Reason:
    The CrossModalMerger attempted to link chunks across documents via cosine
    similarity before chunking was complete.  This design had two fatal flaws:

    1. It required embedding chunks twice (once during merge, once during ingest).
    2. Cross-document linking belongs to the KnowledgeGraph layer, not the
       ingestion pipeline.

Replacement:
    Each document now flows independently through the per-document LangGraph
    IngestGraph (src/pipelines/ingest_graph.py).  Cross-document relationships
    are built lazily by the KnowledgeGraph after all documents are stored.

Do not import from this module.
"""


def CrossModalMerger(*args, **kwargs):
    raise NotImplementedError(
        "CrossModalMerger has been removed.  "
        "Use src/pipelines/ingest_graph.py instead."
    )
