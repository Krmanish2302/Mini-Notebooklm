"""
adaptive_chunker.py

Selects chunk size based on document length:
  short doc  (< 5 pages)  → small chunks  (500 chars)
  medium doc (5-20 pages) → default chunks (1000 chars)
  long doc   (> 20 pages) → large chunks  (2000 chars)
"""
from __future__ import annotations
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .base_chunker import BaseChunker


class AdaptiveChunker(BaseChunker):
    def chunk_documents(self, docs: List[Document]) -> List[Document]:
        n = len(docs)
        if n < 5:
            size, overlap = 500, 50
        elif n <= 20:
            size, overlap = 1000, 200
        else:
            size, overlap = 2000, 300

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size, chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(docs)
        for c in chunks:
            c.metadata["adaptive_chunk_size"] = size
        return chunks