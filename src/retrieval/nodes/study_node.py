"""
study_node.py — LangGraph node: Study mode retrieval.

Retrieves child chunks, reranks them, resolves parents, and queries the SQLite knowledge graph
to extract concept maps (prerequisites, related concepts, examples, contrasts) to guide studying.
"""
from __future__ import annotations
import logging
import os
import re
from typing import List, Dict, Any, Set
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker
from src.retrieval.reorder import reorder_chunks
from src.storage.sqlite_manager import SQLiteManager
from src.generation.llm_registry import LLMRegistry

logger = logging.getLogger(__name__)

_reranker = Reranker()

def _extract_candidates(text: str) -> List[str]:
    candidates = set()
    text_clean = re.sub(r'[\s_\-]+', ' ', text)
    
    # 1. Capitalized n-grams
    cap_words = re.findall(r'\b[A-Z][a-zA-Z0-9]*\b', text_clean)
    for w in cap_words:
        if len(w) >= 3:
            candidates.add(w)
            candidates.add(w.lower())

    # 2. Word n-grams (up to trigrams)
    words = [w.strip() for w in re.split(r'\W+', text_clean) if w.strip()]
    for i in range(len(words)):
        w1 = words[i]
        if len(w1) >= 4:
            candidates.add(w1)
            candidates.add(w1.lower())
        if i < len(words) - 1:
            w2 = f"{words[i]} {words[i+1]}"
            candidates.add(w2)
            candidates.add(w2.lower())
        if i < len(words) - 2:
            w3 = f"{words[i]} {words[i+1]} {words[i+2]}"
            candidates.add(w3)
            candidates.add(w3.lower())
            
    return list(candidates)

_CONCEPT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are an expert tutor. Extract the 1-3 primary technical concepts, definitions, or topics "
               "that are the subject of the user's study query. Output them as a comma-separated list. "
               "Keep them short and exact (e.g. 'Neural Networks, Backpropagation'). "
               "Output ONLY the comma-separated list of concepts, nothing else. If none, output empty."),
    ("human", "Query: {query}\nConcepts:")
])

def _extract_concepts_llm(query: str) -> List[str]:
    """Helper to extract concepts using LLM, with local keyword fallback."""
    try:
        llm = LLMRegistry.get()
        chain = _CONCEPT_PROMPT | llm | StrOutputParser()
        raw = chain.invoke({"query": query})
        concepts = [c.strip() for c in raw.split(",") if c.strip()]
        if concepts:
            return concepts
    except Exception as exc:
        logger.warning("[study_retrieve] LLM concept extraction failed: %s. Falling back.", exc)
    
    # Fallback: simple stop-word removal and capitalization heuristics
    stop = {"what", "is", "are", "was", "were", "explain", "describe", "how", "why", "the", "a", "an", "for", "to", "in", "on", "of", "and"}
    words = re.findall(r"\b\w{4,}\b", query.lower())
    filtered = [w.capitalize() for w in words if w not in stop]
    return filtered[:3]


def normalize_term(t: str) -> str:
    """Normalize terms for robust concept matching."""
    return re.sub(r'[\s_\-]+', ' ', t.lower().strip())


def study_retrieve(state: dict) -> dict:
    try:
        query = state.get("query", "")
        vectorstore_path = state.get("vectorstore_path", "")
        top_k = state.get("top_k", 5)
        source_ids = state.get("source_ids") or None
        use_rerank = state.get("use_rerank", True)

        if not vectorstore_path:
            return {"error": "No vectorstore_path in state", "failed_node": "study_retrieve"}

        # Initialize HybridRetriever for the specific source vectorstore path
        retriever = HybridRetriever(vectorstore_path, top_k=top_k * 3)
        if retriever._ensemble is None:
            retriever._ensemble = retriever._build(top_k)

        # ── 1. Run parallel Dense, Sparse, & Conversational History chunk retrieval ──
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
                logger.warning("[study_retrieve] Failed to initialize history store: %s", e)

        # Collect parallel search results
        dense_docs = dense_future.result() if dense_future is not None else []
        sparse_docs = sparse_future.result() if sparse_future is not None else []
        history_docs = history_future.result() if history_future is not None else []

        # Annotate documents with source_id and source_name
        source_id = os.path.basename(vectorstore_path)
        source_name = source_id
        db = SQLiteManager()
        try:
            source = db.get_source(source_id)
            if source and source.get("name"):
                source_name = source["name"]
        except Exception:
            pass

        for doc in dense_docs + sparse_docs:
            if "source_id" not in doc.metadata:
                doc.metadata["source_id"] = source_id
            doc.metadata["source_name"] = source_name

        # ── 2. Reciprocal Rank Fusion (RRF) for source chunks ─────────────────────────
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

        # ── 3. Rerank source child chunks only ─────────────────────────────────────────
        if use_rerank and len(fused_docs) > 1:
            try:
                child_docs = _reranker.rerank(query, fused_docs, top_n=top_k * 2)
            except Exception as e:
                logger.warning("[study_retrieve] Reranking failed: %s", e)
                child_docs = fused_docs[:top_k * 2]
        else:
            child_docs = fused_docs[:top_k * 2]

        # ── 4. Resolve Parents from SQLite ─────────────────────────────────────────────
        parent_ids = []
        seen_parent_ids = set()

        for doc in child_docs:
            pid = doc.metadata.get("parent_id")
            if pid and pid not in seen_parent_ids:
                seen_parent_ids.add(pid)
                parent_ids.append(pid)

        resolved_parents_list = []
        if parent_ids:
            db_parents = db.get_parents_batch(parent_ids)
            parents_map = {p["parent_id"]: p for p in db_parents}
            for pid in parent_ids:
                if pid in parents_map:
                    resolved_parents_list.append(parents_map[pid])

        # Fallback to child chunks if no parent records exist
        if not resolved_parents_list:
            for i, doc in enumerate(child_docs):
                resolved_parents_list.append({
                    "parent_id": doc.metadata.get("chunk_id") or f"fallback_{i}",
                    "source_id": doc.metadata.get("source_id", "unknown"),
                    "source_type": doc.metadata.get("source_type", "pdf"),
                    "parent_text": doc.page_content,
                    "parent_strategy": "Child fallback",
                    "parent_type": "child_fallback",
                    "range_info": f"Page {doc.metadata.get('page', '')}" if "page" in doc.metadata else "Child chunk",
                    "parent_metadata": doc.metadata,
                    "child_ids": [doc.metadata.get("chunk_id")]
                })

        # ── 5. Localized Parent Reordering (Lost-in-the-Middle) ───────────────────────
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
        reordered_parents = reordered_parents[:top_k]

        # Convert reordered parents to Document format
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
                        "parent_strategy": p["parent_strategy"]
                    }
                )
            )

        # ── 6. SQLite Knowledge Graph Concept Traversal & Normalized Match ─────────────
        concepts = _extract_concepts_llm(query)
        logger.info("[study_retrieve] Extracted query concepts: %s", concepts)
        query_concepts = list(concepts)

        # Build candidate terms to fetch nodes selectively
        candidate_terms = set(concepts)
        candidate_terms.update(_extract_candidates(query))
        for child in child_docs[:top_k]:
            candidate_terms.update(_extract_candidates(child.page_content))

        # Retrieve nodes in DB for normalization/alias match
        try:
            matched_nodes = db.get_graph_nodes_by_names_or_ids(list(candidate_terms))
            logger.info("[study_retrieve] Fetched %d candidate graph nodes from SQLite", len(matched_nodes))
            
            # Prepare normalized targets: (original_node, normalized_name, normalized_id, normalized_aliases)
            normalized_nodes = []
            for n in matched_nodes:
                name = n["name"].strip()
                node_id = n["node_id"].strip()
                meta = n.get("metadata") or {}
                aliases = meta.get("aliases") or []
                
                norm_name = normalize_term(name)
                norm_id = normalize_term(node_id)
                norm_aliases = [normalize_term(a) for a in aliases if a]
                
                normalized_nodes.append({
                    "node": n,
                    "name": norm_name,
                    "node_id": norm_id,
                    "aliases": norm_aliases
                })

            # Hybrid query match using normalized word boundaries
            query_norm = normalize_term(query)
            for entry in normalized_nodes:
                node_name = entry["node"]["name"]
                patterns = [entry["name"], entry["node_id"]] + entry["aliases"]
                found = False
                for pat in patterns:
                    if pat and re.search(r'\b' + re.escape(pat) + r'\b', query_norm):
                        found = True
                        break
                if found and node_name not in concepts:
                    concepts.append(node_name)

            # Controlled seed concept expansion from top child chunks
            for child in child_docs[:top_k]:
                chunk_norm = normalize_term(child.page_content)
                for entry in normalized_nodes:
                    node_name = entry["node"]["name"]
                    patterns = [entry["name"], entry["node_id"]] + entry["aliases"]
                    found = False
                    for pat in patterns:
                        if pat and re.search(r'\b' + re.escape(pat) + r'\b', chunk_norm):
                            found = True
                            break
                    if found and node_name not in concepts:
                        concepts.append(node_name)
            logger.info("[study_retrieve] After normalized word-boundary concept expansion, seeds: %s", concepts)
        
        except Exception as e:
            logger.warning("[study_retrieve] Normalized concept matching failed: %s", e)

        # Retrieve neighbor relations
        graph_context_list = []
        if concepts:
            try:
                edges = db.get_graph_neighbors(concepts)
                logger.info("[study_retrieve] Found %d graph relations", len(edges))
                for edge in edges:
                    graph_context_list.append({
                        "source": edge["source_name"],
                        "target": edge["target_name"],
                        "relation": edge["relation"],
                        "confidence": edge["confidence"],
                        "provenance": edge["provenance"]
                    })
            except Exception as e:
                logger.warning("[study_retrieve] Neighbor lookup failed: %s", e)

        # Fetch current concept descriptions/definitions
        current_concepts_info = []
        try:
            nodes_map = {n["name"].lower(): n for n in matched_nodes}
            nodes_id_map = {n["node_id"].lower(): n for n in matched_nodes}
            for c in query_concepts:
                c_lower = c.lower()
                c_id = c_lower.replace(" ", "_")
                desc = "No description available."
                if c_lower in nodes_map:
                    desc = nodes_map[c_lower].get("metadata", {}).get("description") or desc
                elif c_id in nodes_id_map:
                    desc = nodes_id_map[c_id].get("metadata", {}).get("description") or desc
                current_concepts_info.append({"name": c, "description": desc})
        except Exception as e:
            logger.warning("[study_retrieve] Resolving current concept info failed: %s", e)
            for c in query_concepts:
                current_concepts_info.append({"name": c, "description": "No description available."})

        # Return separately in the state returned
        return {
            "documents": reordered_parent_docs,        # grounded source parents
            "history_docs": history_docs,              # retrieved conversational history
            "graph_context": graph_context_list,       # graph relations / concept links
            "current_concepts": current_concepts_info,  # extracted concepts
            "reordered_parents": reordered_parents,
            "metadata": {
                "dense_count": len(dense_docs),
                "sparse_count": len(sparse_docs),
                "history_count": len(history_docs),
                "rrf_fused": len(fused_docs),
                "reranked": len(child_docs),
                "reordered": len(reordered_parents)
            }
        }

    except Exception as exc:
        logger.exception("[study_retrieve] Failed: %s", exc)
        return {"error": str(exc), "failed_node": "study_retrieve"}
