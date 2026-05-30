"""
hybrid_retriever.py

Hybrid dense + sparse retriever built on LangChain primitives.

Architecture:
  Dense  : ParentDocumentRetriever (FAISS child search → parent doc return)
  Sparse : BM25Retriever (rank_bm25)
  Fusion : EnsembleRetriever (LangChain RRF)

Zero custom vector math — entirely LangChain-native.

Usage:
    from src.retrieval.hybrid_retriever import HybridRetriever

    retriever = HybridRetriever(vectorstore_path="data/vectorstores/rep_001")
    docs = retriever.retrieve("What are the conclusions?", top_k=5)
    # docs → List[Document] of parent chunks (2000 chars each)
"""
from __future__ import annotations
import logging
import os
from typing import List

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

logger = logging.getLogger(__name__)

DENSE_WEIGHT  = float(os.getenv("DENSE_WEIGHT",  "0.7"))
SPARSE_WEIGHT = float(os.getenv("SPARSE_WEIGHT", "0.3"))


class HybridRetriever:
    """
    Hybrid retriever: dense (ParentDocumentRetriever) + sparse (BM25) fused
    via LangChain EnsembleRetriever (Reciprocal Rank Fusion).

    Args:
        vectorstore_path: Path to FAISS + docstore built during ingestion.
        top_k:            Number of documents to return.
        dense_weight:     Weight for dense retriever in RRF (default 0.7).
        sparse_weight:    Weight for BM25 retriever in RRF (default 0.3).
    """

    def __init__(
        self,
        vectorstore_path: str,
        top_k:            int   = 5,
        dense_weight:     float = DENSE_WEIGHT,
        sparse_weight:    float = SPARSE_WEIGHT,
    ):
        self.vectorstore_path = vectorstore_path
        self.top_k            = top_k
        self.dense_weight     = dense_weight
        self.sparse_weight    = sparse_weight
        self._ensemble: EnsembleRetriever | None = None

    def _build(self, top_k: int) -> EnsembleRetriever:
        from src.ingestion.parent_retriever import load_parent_retriever
        from langchain_community.vectorstores import FAISS
        from src.ingestion.embedding.embedding_registry import EmbeddingRegistry

        # ── Dense: ParentDocumentRetriever ─────────────────────────────
        dense = load_parent_retriever(self.vectorstore_path)
        dense.search_kwargs = {"k": top_k * 3}   # fetch 3x, RRF picks best

        # ── Sparse: BM25 built from FAISS stored docs ──────────────────
        embeddings  = EmbeddingRegistry.get()
        vectorstore = FAISS.load_local(
            self.vectorstore_path, embeddings, allow_dangerous_deserialization=True,
        )
        all_docs = list(vectorstore.docstore._dict.values()) if hasattr(vectorstore.docstore, "_dict") else []

        if all_docs:
            bm25 = BM25Retriever.from_documents(all_docs, k=top_k * 3)
        else:
            # Fallback: use dense only if no docs for BM25
            logger.warning("[HybridRetriever] BM25 corpus empty — using dense only")
            dense.search_kwargs = {"k": top_k}
            return dense  # type: ignore[return-value]

        # ── Ensemble (RRF) ─────────────────────────────────────────────
        return EnsembleRetriever(
            retrievers=[dense, bm25],
            weights=[self.dense_weight, self.sparse_weight],
        )

    def retrieve(self, query: str, top_k: int = None) -> List[Document]:
        """
        Retrieve top_k most relevant documents for a query.

        Returns:
            List[Document] — parent chunks (2000 chars) ranked by RRF score.
        """
        k = top_k or self.top_k
        if self._ensemble is None:
            self._ensemble = self._build(k)

        docs = self._ensemble.invoke(query)
        return docs[:k]

    def as_langchain_retriever(self, top_k: int = None):
        """Return as a standard LangChain BaseRetriever for use in LCEL chains."""
        k = top_k or self.top_k
        if self._ensemble is None:
            self._ensemble = self._build(k)
        return self._ensemble