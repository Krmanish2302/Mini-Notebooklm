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

    # Filter to specific sources
    docs = retriever.retrieve("What are the conclusions?", top_k=5,
                              source_ids=["lecture_notes", "paper_02"])
"""
from __future__ import annotations
import logging
import os
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
try:
    from langchain.retrievers import EnsembleRetriever
except ImportError:
    try:
        from langchain_classic.retrievers import EnsembleRetriever
    except ImportError:
        from langchain_community.retrievers import EnsembleRetriever

logger = logging.getLogger(__name__)

DENSE_WEIGHT  = float(os.getenv("DENSE_WEIGHT",  "0.7"))
SPARSE_WEIGHT = float(os.getenv("SPARSE_WEIGHT", "0.3"))


def _load_all_docs_safe(vectorstore) -> List[Document]:
    """
    Extract all docs from a FAISS vectorstore without relying on the
    internal _dict attribute (not part of the public LangChain API).
    """
    # Preferred: _dict shortcut (fast, works in most LangChain versions)
    try:
        docstore = vectorstore.docstore
        if hasattr(docstore, "_dict") and docstore._dict:
            return list(docstore._dict.values())
    except Exception:
        pass

    # Fallback: walk index_to_docstore_id mapping
    try:
        ids  = list(vectorstore.index_to_docstore_id.values())
        docs = []
        for doc_id in ids:
            doc = vectorstore.docstore.search(doc_id)
            if doc and not isinstance(doc, str):
                docs.append(doc)
        return docs
    except Exception as exc:
        logger.warning("[HybridRetriever] Could not extract docs from docstore: %s", exc)
        return []


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
        self._all_docs: List[Document] = []
        self.dense_retriever = None
        self.bm25_retriever = None

    def _build(self, top_k: int) -> EnsembleRetriever:
        from langchain_community.vectorstores import FAISS
        from src.ingestion.embedding.embedding_registry import EmbeddingRegistry

        # Extract source_id from vectorstore_path
        import os
        source_id = os.path.basename(self.vectorstore_path)
        
        # Query database to check embedding model used
        embedding_model = None
        try:
            from src.storage.sqlite_manager import SQLiteManager
            db = SQLiteManager()
            source = db.get_source(source_id)
            if source and "metadata" in source:
                embedding_model = source["metadata"].get("embedding_model")
        except Exception as e:
            logger.warning("[HybridRetriever] Could not fetch embedding model for source %s: %s", source_id, e)

        embeddings = EmbeddingRegistry.get(embedding_model)

        # ── Load FAISS once and share across dense + BM25 ──────────────
        vectorstore = FAISS.load_local(
            self.vectorstore_path, embeddings, allow_dangerous_deserialization=True,
        )

        # ── Dense: FAISS as retriever ──────────────────────────────────
        dense = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k * 3},
        )

        # ── Sparse: BM25 built from FAISS stored docs ─────────────────
        all_docs = _load_all_docs_safe(vectorstore)
        self._all_docs = all_docs

        self.dense_retriever = dense
        if all_docs:
            self.bm25_retriever = BM25Retriever.from_documents(all_docs, k=top_k * 3)
        else:
            self.bm25_retriever = None

        if self.bm25_retriever is None:
            logger.warning("[HybridRetriever] BM25 corpus empty — using dense only")
            dense.search_kwargs = {"k": top_k}
            return dense  # type: ignore[return-value]

        return EnsembleRetriever(
            retrievers=[self.dense_retriever, self.bm25_retriever],
            weights=[self.dense_weight, self.sparse_weight],
        )

    def retrieve(
        self,
        query:      str,
        top_k:      int = None,
        source_ids: Optional[List[str]] = None,
    ) -> List[Document]:
        """
        Retrieve top_k most relevant documents for a query.

        Args:
            query:      Search query string.
            top_k:      Number of results (default: self.top_k).
            source_ids: If provided, only return docs whose metadata
                        'source_id' is in this list.
        """
        k = top_k or self.top_k

        if self._ensemble is None:
            self._ensemble = self._build(k)

        docs = self._ensemble.invoke(query)

        # Post-filter by source_ids if requested
        if source_ids:
            sid_set = set(source_ids)
            docs = [d for d in docs if d.metadata.get("source_id") in sid_set]

        return docs[:k]

    def as_langchain_retriever(self, top_k: int = None):
        """Return as a standard LangChain BaseRetriever for use in LCEL chains."""
        k = top_k or self.top_k
        if self._ensemble is None:
            self._ensemble = self._build(k)
        return self._ensemble
