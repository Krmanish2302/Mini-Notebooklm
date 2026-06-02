"""
build_context_node.py — LangGraph node: build final context string + error handler.
"""
from __future__ import annotations
import logging
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def format_study_graph_context(graph_context: list, current_concepts: list) -> str:
    prereqs = []
    related = []
    examples_contrasts = []
    
    seen_edges = set()
    unique_edges = []
    for rel in graph_context:
        edge_key = (rel["source"].lower(), rel["target"].lower(), rel["relation"].lower())
        if edge_key not in seen_edges:
            seen_edges.add(edge_key)
            unique_edges.append(rel)

    for rel in unique_edges:
        src = rel["source"]
        tgt = rel["target"]
        relation = rel["relation"].lower()
        conf = rel.get("confidence", 1.0)
        prov = rel.get("provenance") or {}
        prov_str = ""
        if prov:
            prov_src = prov.get("source_id", "")
            prov_page = prov.get("page", "")
            prov_str = f" (Source: {prov_src}" + (f", p. {prov_page}" if prov_page else "") + ")"
        
        # Classify relation types
        if relation in ("prerequisite_of", "depends_on", "prereq"):
            prereqs.append(f"- **{src}** is a prerequisite for **{tgt}** [Confidence: {conf}]{prov_str}")
        elif relation in ("contrast_with", "contrasts", "opposes", "difference"):
            examples_contrasts.append(f"- **{src}** contrasts with **{tgt}** [Confidence: {conf}]{prov_str}")
        elif relation in ("example_of", "example", "instance"):
            examples_contrasts.append(f"- **{src}** is an example of **{tgt}** [Confidence: {conf}]{prov_str}")
        else:
            related.append(f"- **{src}** is related to **{tgt}** ({rel['relation']}) [Confidence: {conf}]{prov_str}")
            
    current = []
    for cc in current_concepts:
        name = cc.get("name", "")
        desc = cc.get("description", "")
        current.append(f"- **{name}**: {desc}")

    blocks = []
    if prereqs:
        blocks.append("### Prerequisite Concepts\n" + "\n".join(prereqs))
    if current:
        blocks.append("### Current Concept(s)\n" + "\n".join(current))
    if related:
        blocks.append("### Related Concepts\n" + "\n".join(related))
    if examples_contrasts:
        blocks.append("### Examples & Contrasts\n" + "\n".join(examples_contrasts))
        
    if blocks:
        return "## Concept Knowledge Map (SQLite Study Graph)\n\n" + "\n\n".join(blocks)
    return ""


def build_context(state: dict) -> dict:
    try:
        mode = state.get("mode", "chat")

        if mode == "study":
            source_parents = state.get("documents", []) or state.get("reordered_docs", [])
            history_docs = state.get("history_docs", [])
            graph_context = state.get("graph_context", [])
            current_concepts = state.get("current_concepts", [])
            
            from src.retrieval.context_builder import MAX_CONTEXT_CHARS, ContextBuilder
            remaining_budget = MAX_CONTEXT_CHARS
            
            graph_str = format_study_graph_context(graph_context, current_concepts)
            if len(graph_str) > 3000:
                graph_str = graph_str[:3000]
            remaining_budget -= len(graph_str)
            
            history_str = ""
            if history_docs and remaining_budget > 1000:
                hist_budget = min(3000, remaining_budget)
                history_str = "## Retrieved Conversation History\n\n" + ContextBuilder(max_context_chars=hist_budget).build(history_docs, query="")
                remaining_budget -= len(history_str)
                
            source_str = ""
            if source_parents and remaining_budget > 1000:
                source_str = "## Grounded Source Documents\n\n" + ContextBuilder(max_context_chars=remaining_budget).build(source_parents, query="")
                remaining_budget -= len(source_str)
                
            parts = []
            if graph_str:
                parts.append(graph_str)
            if history_str:
                parts.append(history_str)
            if source_str:
                parts.append(source_str)
                
            context = "\n\n".join(parts)
            
            combined_docs = []
            if graph_str:
                combined_docs.append(Document(
                    page_content=graph_str,
                    metadata={"source_id": "SQLite Study Graph"}
                ))
            combined_docs.extend(history_docs)
            combined_docs.extend(source_parents)
            
            logger.info("[build_context] Context built for Study mode, docs=%d, chars=%d", len(combined_docs), len(context))
            existing_meta = state.get("metadata") or {}
            new_meta = {
                **existing_meta,
                "num_docs": len(combined_docs),
                "context_chars": len(context)
            }
            return {
                "context": context,
                "documents": combined_docs,
                "metadata": new_meta
            }

        # Chat / Deep Research modes
        docs = (
            state.get("reordered_docs")
            or state.get("reranked_docs")
            or state.get("documents", [])
        )

        from src.retrieval.context_builder import ContextBuilder
        context = ContextBuilder().build(docs, query="")

        # ── Format knowledge graph context for other modes if present ─────────
        graph_context = state.get("graph_context", [])
        graph_str = ""
        if mode == "study" and graph_context: # fallback (though handled above)
            pass

        logger.info("[build_context] Context built from %d docs (%d chars), mode=%s", len(docs), len(context), mode)
        existing_meta = state.get("metadata") or {}
        new_meta = {
            **existing_meta,
            "num_docs": len(docs),
            "context_chars": len(context)
        }
        return {
            "context":   context,
            "documents": docs,
            "metadata":  new_meta,
        }

    except Exception as exc:
        logger.exception("[build_context] Failed: %s", exc)
        return {"error": str(exc), "failed_node": "build_context"}


def handle_error(state: dict) -> dict:
    logger.error(
        "[retrieval] FAILED at node='%s': %s",
        state.get("failed_node", "unknown"),
        state.get("error", "Unknown error"),
    )
    return {
        "context":   "",
        "documents": [],
        "metadata":  {"num_docs": 0, "context_chars": 0},
    }
