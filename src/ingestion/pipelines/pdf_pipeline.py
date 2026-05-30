"""
pdf_pipeline.py

LangChain PyMuPDFLoader wrapper.
Returns List[Document] with per-page metadata.
Used by loader_node.py directly — this class is for standalone use.
"""
from __future__ import annotations
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader


class PDFPipeline:
    @staticmethod
    def process(file_path: str, source_id: str) -> List[Document]:
        """
        Load PDF using PyMuPDFLoader.

        Returns:
            List[Document] with metadata:
              page, source, source_id, source_type
        """
        loader = PyMuPDFLoader(file_path)
        docs   = loader.load()
        for doc in docs:
            doc.metadata["source_id"]   = source_id
            doc.metadata["source_type"] = "pdf"
        return docs