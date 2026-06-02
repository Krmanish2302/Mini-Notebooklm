"""
semantic_chunker.py

Wraps LangChain SemanticChunker (langchain-experimental).
Splits on semantic similarity breakpoints using embeddings.
Requires OPENAI_API_KEY or swap embeddings for HuggingFace.
"""
from __future__ import annotations
import os
from typing import List
from langchain_core.documents import Document
from .base_chunker import BaseChunker


class SemanticChunker(BaseChunker):
    def __init__(
        self,
        breakpoint_threshold_type:   str = "percentile",
        breakpoint_threshold_amount: int = 90,
        embedding_provider:          str = os.getenv("EMBEDDING_PROVIDER", "huggingface"),
    ):
        self.threshold_type   = breakpoint_threshold_type
        self.threshold_amount = breakpoint_threshold_amount
        self.embedding_provider = embedding_provider

    def _get_embeddings(self):
        if self.embedding_provider == "openai":
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(model="text-embedding-3-small")
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    def chunk_documents(self, docs: List[Document]) -> List[Document]:
        from langchain_experimental.text_splitter import SemanticChunker as _SC
        chunker = _SC(
            self._get_embeddings(),
            breakpoint_threshold_type=self.threshold_type,
            breakpoint_threshold_amount=self.threshold_amount,
        )
        return chunker.split_documents(docs)