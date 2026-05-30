"""page_chunker.py — One chunk per page (already split by loader)."""
from __future__ import annotations
from typing import List
from langchain_core.documents import Document
from .base_chunker import BaseChunker


class PageChunker(BaseChunker):
    def chunk_documents(self, docs: List[Document]) -> List[Document]:
        for i, doc in enumerate(docs):
            doc.metadata["chunk_index"] = i
            doc.metadata["chunker"]     = "page"
        return docs