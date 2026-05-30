"""
recursive_chunker.py

Wraps LangChain RecursiveCharacterTextSplitter.
Used directly in chunking_node.py for the main pipeline.
This class is available for standalone use outside the pipeline.
"""
from __future__ import annotations
import os
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .base_chunker import BaseChunker


class RecursiveChunker(BaseChunker):
    def __init__(
        self,
        chunk_size:    int       = int(os.getenv("CHUNK_SIZE",    "1000")),
        chunk_overlap: int       = int(os.getenv("CHUNK_OVERLAP", "200")),
        separators:    List[str] = None,
    ):
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators or ["\n\n", "\n", ". ", " ", ""],
            add_start_index=True,
        )

    def chunk_documents(self, docs: List[Document]) -> List[Document]:
        return self._splitter.split_documents(docs)