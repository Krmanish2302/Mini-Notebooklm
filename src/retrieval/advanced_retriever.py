"""
advanced_retriever.py

High-level retriever that composes:
  HybridRetriever      (dense + sparse, RRF)
  SubQueryDecomposer   (query expansion)
  Reranker             (cross-encoder reranking)
  ContextBuilder       (format context string)

This is the single class called by generation/ and API layers.

Usage:
    from src.retrieval.advanced_retriever import AdvancedRetriever

    retriever = AdvancedRetriever(vectorstore_path="data/vectorstores/rep_001")
    result = retriever.retrieve("What is the conclusion?")
    print(result["context"])    # formatted string → feed to LLM
    print(result["documents"])  # List[Document]
"""
from __future__ import annotations
import logging
import os
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from .hybrid_retriever      import HybridRetriever
from .query_expander        import SubQueryDecomposer, MultiQueryExpander
from .reranker              import Reranker
from .context_builder       import ContextBuilder

logger = logging.getLogger(__name__)

TOP_K          = int(os.getenv("RETRIEVAL_TOP_K",    "5"))
EXPAND_QUERIES = os.getenv("EXPAND_QUERIES", "true").lower() == "true"
USE_RERANK     = os.getenv("USE_RERANK",     "true").lower() == "true"


class AdvancedRetriever:
    """
    Full retrieval pipeline: expand → hybrid retrieve → rerank → build context.

    Args:
        vectorstore_path: Path to FAISS + docstore from ingestion.
        top_k:            Number of final documents to return.
        expand_queries:   Whether to decompose query into sub-queries.
        use_rerank:       Whether to apply cross-encoder reranking.
        use_compression:  Whether to apply LLM contextual compression.
    """

    def __init__(
        self,
        vectorstore_path: str,
        top_k:            int  = TOP_K,
        expand_queries:   bool = EXPAND_QUERIES,
        use_rerank:       bool = USE_RERANK,
        use_compression:  bool = False,
    ):
        self.vectorstore_path = vectorstore_path
        self.top_k            = top_k
        self.expand_queries   = expand_queries
        self.use_rerank       = use_rerank
        self.use_compression  = use_compression

        self._hybrid    = HybridRetriever(vectorstore_path, top_k=top_k * 3)
        self._decomposer = SubQueryDecomposer(n=3, use_llm=expand_queries)
        self._expander   = MultiQueryExpander(n=3, use_llm=expand_queries)
        self._reranker   = Reranker()
        self._builder    = ContextBuilder()

    def retrieve(self, query: str) -> Dict[str, Any]:
        """
        Full retrieval pipeline.

        Returns:
            {
                "context":   str,            # formatted for LLM prompt
                "documents": List[Document], # raw parent chunks
                "queries":   List[str],      # expanded queries used
            }
        """
        # 1. Query expansion
        queries = [query]
        if self.expand_queries:
            queries = self._decomposer.decompose(query)
            logger.info("[AdvancedRetriever] Expanded to %d queries", len(queries))

        # 2. Hybrid retrieve for each query, deduplicate
        seen: set = set()
        all_docs: List[Document] = []
        for q in queries:
            for doc in self._hybrid.retrieve(q, top_k=self.top_k * 2):
                key = hash(doc.page_content[:200])
                if key not in seen:
                    seen.add(key)
                    all_docs.append(doc)

        logger.info("[AdvancedRetriever] Retrieved %d unique docs", len(all_docs))

        # 3. Rerank
        if self.use_rerank and all_docs:
            all_docs = self._reranker.rerank(query, all_docs, top_n=self.top_k)

        # 4. Contextual compression (optional, costs LLM calls)
        if self.use_compression and all_docs:
            from .contextual_compressor import ContextualCompressor
            all_docs = ContextualCompressor().compress(query, all_docs)

        # 5. Final trim
        final_docs = all_docs[:self.top_k]

        # 6. Build context
        context = self._builder.build(final_docs, query)

        return {
            "context":   context,
            "documents": final_docs,
            "queries":   queries,
        }

    def as_langchain_retriever(self):
        """Return underlying HybridRetriever as standard LangChain BaseRetriever."""
        return self._hybrid.as_langchain_retriever(top_k=self.top_k)