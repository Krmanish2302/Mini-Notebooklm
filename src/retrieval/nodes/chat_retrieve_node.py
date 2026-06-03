"""
chat_retrieve_node.py — LangGraph node: Chat mode retrieval.

Performs parallel Dense (FAISS) similarity search, Sparse (BM25) search, and semantic history retrieval.
Applies RRF, reranks source chunks, reorders source chunks (lost-in-the-middle), and prepends history.
"""
from __future__ import annotations
import logging
import os
from typing import List, Dict, Any
from langchain_core.documents import Document
from src.retrieval.reranker import Reranker

logger = logging.getLogger(__name__)

_reranker = Reranker()


def chat_retrieve(state: dict) -> dict:

    try:
        query = state.get("query", "")
        vectorstore_path = state.get("vectorstore_path", "")
        top_k = state.get("top_k", 5)
        source_ids = state.get("source_ids") or None

        if not vectorstore_path:
            return {"error": "No vectorstore_path in state", "failed_node": "chat_retrieve"}

        # Initialize HybridRetriever for the specific source vectorstore path
        from src.retrieval.hybrid_retriever import HybridRetriever
        retriever = HybridRetriever(vectorstore_path, top_k=top_k * 3)
        if retriever._ensemble is None:
            retriever._ensemble = retriever._build(top_k)

        # Run dense, sparse, and history retrieval in parallel
        from concurrent.futures import ThreadPoolExecutor
        dense_future = None
        sparse_future = None
        history_future = None

        with ThreadPoolExecutor(max_workers=3) as executor:
            if retriever.dense_retriever is not None:
                dense_future = executor.submit(retriever.dense_retriever.invoke, query)
            if retriever.bm25_retriever is not None:
                sparse_future = executor.submit(retriever.bm25_retriever.invoke, query)
            
            # History retrieval
            try:
                from src.storage.rag_history_store import RAGHistoryStore
                from src.storage.sqlite_manager import SQLiteManager
                from src.ingestion.embedding.embedding_registry import EmbeddingRegistry
                db = SQLiteManager()
                embedder = EmbeddingRegistry.get()
                history_store = RAGHistoryStore(db, embedder)
                history_future = executor.submit(
                    history_store.retrieve_history_docs,
                    session_id="default",
                    current_query=query,
                    top_k=2
                )
            except Exception as e:
                logger.warning("[chat_retrieve] Failed to initialize history store: %s", e)

        # Collect results
        dense_docs = dense_future.result() if dense_future is not None else []
        sparse_docs = sparse_future.result() if sparse_future is not None else []
        history_docs = history_future.result() if history_future is not None else []

        # Annotate documents with source_id and source_name
        source_id = os.path.basename(vectorstore_path)
        source_name = source_id
        try:
            from src.storage.sqlite_manager import SQLiteManager
            db = SQLiteManager()
            source = db.get_source(source_id)
            if source and source.get("name"):
                source_name = source["name"]
        except Exception:
            pass

        for doc in dense_docs:
            if "source_id" not in doc.metadata:
                doc.metadata["source_id"] = source_id
            doc.metadata["source_name"] = source_name
        for doc in sparse_docs:
            if "source_id" not in doc.metadata:
                doc.metadata["source_id"] = source_id
            doc.metadata["source_name"] = source_name

        # Reciprocal Rank Fusion (RRF) for dense and sparse results
        def get_cid(d):
            return d.metadata.get("chunk_id") or d.metadata.get("id") or str(hash(d.page_content))

        dense_ids, seen_dense = [], set()
        for doc in dense_docs:
            cid = get_cid(doc)
            if cid not in seen_dense:
                seen_dense.add(cid)
                dense_ids.append(cid)

        sparse_ids, seen_sparse = [], set()
        for doc in sparse_docs:
            cid = get_cid(doc)
            if cid not in seen_sparse:
                seen_sparse.add(cid)
                sparse_ids.append(cid)

        RRF_K = 60
        scores = {}
        for rank, cid in enumerate(dense_ids):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, cid in enumerate(sparse_ids):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)

        fused_cids = sorted(scores, key=lambda cid: scores[cid], reverse=True)[:top_k * 3]
        doc_map = {get_cid(d): d for d in dense_docs + sparse_docs}
        fused_docs = [doc_map[cid] for cid in fused_cids if cid in doc_map]

        # Rerank only the source child chunks
        if len(fused_docs) > 1:
            try:
                reranked_docs = _reranker.rerank(query, fused_docs, top_n=top_k)
            except Exception as e:
                logger.warning("[chat_retrieve] Reranking failed: %s", e)
                reranked_docs = fused_docs[:top_k]
        else:
            reranked_docs = fused_docs[:top_k]

        # Apply lost-in-the-middle reordering on source chunks only
        try:
            from src.retrieval.reorder import reorder_chunks
            chunks_with_scores = []
            for doc in reranked_docs:
                score = doc.metadata.get("relevance_score", doc.metadata.get("score", 0.0))
                try:
                    score = float(score)
                except (ValueError, TypeError):
                    score = 0.0
                chunks_with_scores.append((doc, score))
            reordered_docs = reorder_chunks(chunks_with_scores)
        except Exception as e:
            logger.warning("[chat_retrieve] Reordering failed: %s", e)
            reordered_docs = reranked_docs

        # Merge history chunks and reordered source chunks
        combined_docs = history_docs + reordered_docs

        metadata = {
            "dense_count": len(dense_docs),
            "sparse_count": len(sparse_docs),
            "history_count": len(history_docs),
            "rrf_fused": len(fused_docs),
            "reranked": len(reranked_docs),
            "reordered": len(reordered_docs)
        }

        logger.info(
            "[chat_retrieve] Chat retrieval done: dense=%d sparse=%d history=%d fused=%d reranked=%d reordered=%d",
            len(dense_docs), len(sparse_docs), len(history_docs), len(fused_docs), len(reranked_docs), len(reordered_docs)
        )

        return {
            "documents": combined_docs,
            "reordered_docs": combined_docs,
            "reranked_docs": combined_docs,
            "metadata": metadata
        }

    except Exception as exc:
        logger.exception("[chat_retrieve] Failed: %s", exc)
        return {"error": str(exc), "failed_node": "chat_retrieve"}
