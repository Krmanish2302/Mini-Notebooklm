"""
semantic_chunker.py

Chunks text based on semantic similarity between sentences.
Uses langchain-experimental SemanticChunker with HuggingFace embeddings.

Fix: updated import path for HuggingFaceEmbeddings
     (langchain.embeddings → langchain_community.embeddings)
"""
from langchain_experimental.text_splitter import SemanticChunker as LCSemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings  # fixed import
from .base_chunker import BaseChunker
from typing import List, Dict, Any


class SemanticChunker(BaseChunker):
    """Chunks based on semantic similarity using sentence embeddings."""

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        breakpoint_threshold: float = 85.0,
    ):
        embeddings = HuggingFaceEmbeddings(
            model_name=f"sentence-transformers/{embedding_model}"
            if "/" not in embedding_model
            else embedding_model
        )
        self.splitter = LCSemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=breakpoint_threshold,
        )

    def chunk(
        self, content: str, metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        metadata = metadata or {}
        docs = self.splitter.create_documents([content])
        return [
            {
                "id": f"{metadata.get('source_id', 'unknown')}_chunk_{i}",
                "content": doc.page_content,
                "metadata": {**metadata, "chunk_index": i},
                "modality": metadata.get("modality", "text"),
            }
            for i, doc in enumerate(docs)
        ]

    def get_strategy_name(self) -> str:
        return "semantic"
