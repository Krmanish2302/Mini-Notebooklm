"""paragraph_chunker.py — Split on double newline (paragraph boundary)."""
from __future__ import annotations
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .base_chunker import BaseChunker


class ParagraphChunker(BaseChunker):
    def __init__(self, max_chunk_size: int = 1500, overlap: int = 100):
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk_documents(self, docs: List[Document]) -> List[Document]:
        return self._splitter.split_documents(docs)