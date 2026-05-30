"""
raptor_builder.py — DEPRECATED stub.
Replaced by ParentDocumentRetriever. See parent_retriever.py.
"""
import logging
logger = logging.getLogger(__name__)

def build_raptor_tree(*args, **kwargs) -> None:
    logger.warning(
        "build_raptor_tree() is deprecated. "
        "Use src.ingestion.parent_retriever.build_parent_retriever() instead."
    )