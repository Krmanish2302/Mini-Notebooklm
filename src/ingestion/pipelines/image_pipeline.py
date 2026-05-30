"""
image_pipeline.py — Image text extraction via UnstructuredImageLoader.
Falls back to empty document with warning if unstructured not installed.
"""
from __future__ import annotations
import logging
from typing import List
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class ImagePipeline:
    @staticmethod
    def process(file_path: str, source_id: str) -> List[Document]:
        try:
            from langchain_community.document_loaders import UnstructuredImageLoader
            docs = UnstructuredImageLoader(file_path).load()
        except ImportError:
            logger.warning("[ImagePipeline] unstructured not installed. Returning empty.")
            return []
        for doc in docs:
            doc.metadata["source_id"]   = source_id
            doc.metadata["source_type"] = "image"
        return docs