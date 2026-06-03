"""
deep_research_node.py — LangGraph node: Deep Research mode retrieval.

Performs conditional query expansion, hybrid parallel retrieval over child chunks,
reranking, parent resolution from SQLite, grounding evidence mapping,
lost-in-the-middle reordering on parents, and merges with history.
"""
from __future__ import annotations
import logging
import os
import json
from typing import List, Set, Dict, Any, Tuple
from langchain_core.documents import Document

from src.retrieval.query_expander import SubQueryDecomposer
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker
from src.retrieval.reorder import reorder_chunks
from src.storage.sqlite_manager import SQLiteManager

logger = logging.getLogger(__name__)

_reranker = Reranker()


def should_expand_query(query: str) -> bool:
    """
    Heuristics to decide if a query warrants alternate sub-queries:
    - exploratory keywords (compare, differences, overview, summary, deep dive)
    - punctuation/conjunction markers suggesting multiple parts
    - query length
    """
    q_lower = query.lower()
    
    # 1. Exploratory / Research Keywords
    research_words = {
        "compare", "comparison", "difference", "differences", "contrast", 
        "overview", "summary", "summarize", "deep dive", "comprehensive", 
        "detailed", "history", "evolution", "background", "analyze", "analysis",
        "versus", "vs", "explain all", "relationship", "relation"
    }
    if any(word in q_lower for word in research_words):
        return True
        
    # 2. Multi-part indicators (multiple questions or query clauses)
    if q_lower.count("?") > 1 or q_lower.count(";") > 0:
        return True
        
    # 3. Long queries with conjunctions suggesting multiple clauses/sub-questions
    conjunctions = ["and also", "as well as", "along with", "in addition to"]
    if any(conj in q_lower for conj in conjunctions):
        return True
        
    # 4. Long queries suggest detail and multi-part intent
    if len(q_lower.split()) > 15:
        return True
        
    return False

def deep_research_retrieve(state: dict) -> dict:
    try:
        query = state.get("query", "")
        vectorstore_path = state.get("vectorstore_path", "")
        top_k = state.get("top_k", 5)
        source_ids = state.get("source_ids") or None
        do_expand = state.get("do_expand", True)
        use_rerank = state.get("use_rerank", True)

        if not vectorstore_path:
            return {"error": "No vectorstore_path in state", "failed_node": "deep_research_retrieve"}

        # ── 1. Conditional Query Expansion ──────────────────────────────────
        queries = [query]
        if do_expand and should_expand_query(query):
            logger.info("[deep_research_retrieve] Triggering conditional query expansion")
            try:
                decomposer = SubQueryDecomposer(n=3, use_llm=True)
                sub_queries = decomposer.decompose(query)
                queries = list(dict.fromkeys([query] + sub_queries))
                logger.info("[deep_research_retrieve] Expanded queries: %s", queries)
            except Exception as e:
                logger.warning("[deep_research_retrieve] Query decomposition failed: %s", e)
        else:
            logger.info("[deep_research_retrieve] Skipping query expansion")

        # Initialize HybridRetriever
        retriever = HybridRetriever(vectorstore_path, top_k=top_k * 3)
        if retriever._ensemble is None:
            retriever._ensemble = retriever._build(top_k)

        # ── 2. Run Parallel Dense, Sparse & History search ───────────────────
        from concurrent.futures import ThreadPoolExecutor
        dense_futures = []
        sparse_futures = []
        history_future = None

        with ThreadPoolExecutor(max_workers=8) as executor:
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
                logger.warning("[deep_research_retrieve] History store init failed: %s", e)

            # Source retrieval for all sub-queries
            for q in queries:
                if retriever.dense_retriever is not None:
                    dense_futures.append(executor.submit(retriever.dense_retriever.invoke, q))
                if retriever.bm25_retriever is not None:
                    sparse_futures.append(executor.submit(retriever.bm25_retriever.invoke, q))

        # Collect results
        dense_docs = []
        for fut in dense_futures:
            try:
                dense_docs.extend(fut.result())
            except Exception as e:
                logger.warning("[deep_research_retrieve] Parallel dense search failed: %s", e)

        sparse_docs = []
        for fut in sparse_futures:
            try:
                sparse_docs.extend(fut.result())
            except Exception as e:
                logger.warning("[deep_research_retrieve] Parallel sparse search failed: %s", e)

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

        for doc in dense_docs + sparse_docs:
            if "source_id" not in doc.metadata:
                doc.metadata["source_id"] = source_id
            doc.metadata["source_name"] = source_name

        # ── 3. RRF Fusion (dense and sparse only) ───────────────────────────
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

        # ── 4. Rerank source child chunks only ─────────────────────────────
        if use_rerank and len(fused_docs) > 1:
            try:
                child_docs = _reranker.rerank(query, fused_docs, top_n=top_k * 2)
            except Exception as e:
                logger.warning("[deep_research_retrieve] Reranking failed: %s", e)
                child_docs = fused_docs[:top_k * 2]
        else:
            child_docs = fused_docs[:top_k * 2]

        # ── 5. Resolve Parents and map Grounding Evidence ───────────────────
        db = SQLiteManager()
        parent_ids = []
        seen_parent_ids = set()
        parent_to_children = {}

        for doc in child_docs:
            pid = doc.metadata.get("parent_id")
            if pid:
                parent_to_children.setdefault(pid, []).append(doc)
                if pid not in seen_parent_ids:
                    seen_parent_ids.add(pid)
                    parent_ids.append(pid)

        logger.info("[deep_research_retrieve] Resolving %d unique parent IDs", len(parent_ids))
        
        resolved_parents_list = []
        if parent_ids:
            db_parents = db.get_parents_batch(parent_ids)
            parents_map = {p["parent_id"]: p for p in db_parents}
            
            for pid in parent_ids:
                if pid in parents_map:
                    p = parents_map[pid]
                    # Map the supporting children details as grounding evidence
                    supporting = parent_to_children[pid]
                    p_metadata = p.get("parent_metadata") or {}
                    if isinstance(p_metadata, str):
                        try:
                            p_metadata = json.loads(p_metadata)
                        except Exception:
                            p_metadata = {}
                    
                    p_metadata["supporting_children"] = [
                        {
                            "child_id": c.metadata.get("chunk_id"),
                            "text_snippet": c.page_content[:200] + "...",
                            "page_number": c.metadata.get("page_number") or c.metadata.get("page"),
                            "relevance_score": c.metadata.get("relevance_score", c.metadata.get("score", 0.0))
                        }
                        for c in supporting
                    ]
                    p["parent_metadata"] = p_metadata
                    resolved_parents_list.append(p)

        # Fallback to treating child chunks as parents if none found
        if not resolved_parents_list:
            logger.warning("[deep_research_retrieve] No parent records found in SQLite, falling back to children")
            for i, doc in enumerate(child_docs):
                resolved_parents_list.append({
                    "parent_id": doc.metadata.get("chunk_id") or f"fallback_{i}",
                    "source_id": doc.metadata.get("source_id", "unknown"),
                    "source_type": doc.metadata.get("source_type", "pdf"),
                    "parent_text": doc.page_content,
                    "parent_strategy": "Child fallback",
                    "parent_type": "child_fallback",
                    "range_info": f"Page {doc.metadata.get('page', '')}" if "page" in doc.metadata else "Child chunk",
                    "parent_metadata": {**doc.metadata, "supporting_children": [{"child_id": doc.metadata.get("chunk_id"), "text_snippet": doc.page_content[:200] + "..."}]},
                    "child_ids": [doc.metadata.get("chunk_id")]
                })

        # ── 6. Lost-in-the-Middle Reordering of Parents ─────────────────────
        child_score_map = {}
        for doc in child_docs:
            score = float(doc.metadata.get("relevance_score", doc.metadata.get("score", 0.0)))
            pid = doc.metadata.get("parent_id")
            cid = doc.metadata.get("chunk_id")
            if pid:
                child_score_map[pid] = max(child_score_map.get(pid, -9999.0), score)
            if cid:
                child_score_map[cid] = max(child_score_map.get(cid, -9999.0), score)

        parents_with_scores = [
            (p, child_score_map.get(p["parent_id"], 0.0)) 
            for p in resolved_parents_list
        ]
        reordered_parents = reorder_chunks(parents_with_scores)
        
        # Limit to top_k parents to stay within budget
        reordered_parents = reordered_parents[:top_k]

        # Convert to Documents for build_context
        reordered_parent_docs = []
        for p in reordered_parents:
            reordered_parent_docs.append(
                Document(
                    page_content=p["parent_text"],
                    metadata={
                        "source_id": p["source_id"],
                        "page": p.get("range_info") or p.get("parent_metadata", {}).get("pages", ""),
                        "parent_id": p["parent_id"],
                        "parent_type": p["parent_type"],
                        "parent_strategy": p["parent_strategy"],
                        "supporting_children": p.get("parent_metadata", {}).get("supporting_children", [])
                    }
                )
            )

        # ── 7. Merge history turns ──────────────────────────────────────────
        combined_docs = history_docs + reordered_parent_docs

        metadata = {
            "dense_count": len(dense_docs),
            "sparse_count": len(sparse_docs),
            "history_count": len(history_docs),
            "rrf_fused": len(fused_docs),
            "reranked": len(child_docs),
            "reordered": len(reordered_parents)
        }

        logger.info(
            "[deep_research_retrieve] Done: dense=%d sparse=%d history=%d fused=%d child_reranked=%d parents_reordered=%d",
            len(dense_docs), len(sparse_docs), len(history_docs), len(fused_docs), len(child_docs), len(reordered_parent_docs)
        )

        return {
            "reordered_parents": reordered_parents,
            "reordered_docs": combined_docs,
            "documents": combined_docs,
            "expanded_queries": queries,
            "metadata": metadata
        }

    except Exception as exc:
        logger.exception("[deep_research_retrieve] Failed: %s", exc)
        return {"error": str(exc), "failed_node": "deep_research_retrieve"}
