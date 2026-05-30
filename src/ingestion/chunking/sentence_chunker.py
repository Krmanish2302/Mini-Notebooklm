"""sentence_chunker.py — Token-aware sentence chunker via LangChain."""
from __future__ import annotations
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from .base_chunker import BaseChunker


class SentenceChunker(BaseChunker):
    def __init__(self, chunk_overlap: int = 0, tokens_per_chunk: int = 256):
        self._splitter = SentenceTransformersTokenTextSplitter(
            chunk_overlap=chunk_overlap,
            tokens_per_chunk=tokens_per_chunk,
        )

    def chunk_documents(self, docs: List[Document]) -> List[Document]:
        return self._splitter.split_documents(docs)