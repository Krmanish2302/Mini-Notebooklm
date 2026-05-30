"""
raptor_builder.py  —  DEPRECATED

RAPTOR has been replaced by ParentDocumentRetriever with persistent
LocalFileStore.  This file is kept as a stub to avoid breaking any
external imports, but does nothing.

See: src/ingestion/parent_retriever.py
"""
import logging

logger = logging.getLogger(__name__)


def build_raptor_tree(*args, **kwargs) -> None:  # noqa: ANN002
    logger.warning(
        "build_raptor_tree() is deprecated and does nothing. "
        "Use src.ingestion.parent_retriever.build_parent_retriever() instead."
    )
