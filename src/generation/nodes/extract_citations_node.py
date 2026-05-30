"""
extract_citations_node.py — LangGraph node: extract & resolve inline citations + final assembly.
"""
from __future__ import annotations
import logging

logger = logging.getLogger(__name__)


def extract_citations(state: dict) -> dict:
    try:
        answer     = state.get("answer", "")
        documents  = state.get("documents", [])

        from src.generation.citation_extractor import CitationExtractor
        from src.generation.response_generator import ResponseGenerator

        # Build context_chunks list (uniform dict format)
        chunks = []
        for i, doc in enumerate(documents, 1):
            if hasattr(doc, "page_content"):
                content = doc.page_content
                meta    = doc.metadata or {}
            else:
                content = doc.get("content", "")
                meta    = {k: v for k, v in doc.items() if k != "content"}
            chunks.append({
                "citation_label": f"S{i}",
                "content":        content,
                **meta,
            })

        # Extract citations
        extractor   = CitationExtractor(chunks)
        citations   = extractor.extract(answer)
        validation  = extractor.validate(answer)

        # Assemble final response
        generator   = ResponseGenerator(chunks)
        assembled   = generator.assemble(
            raw_llm_output=state.get("raw_output", answer),
            query=state.get("query", ""),
        )

        # Merge: prefer assembled answer (it normalizes citations) but keep node-parsed follow_ups
        return {
            "answer":          assembled["answer"],
            "citations":       citations,
            "follow_ups":      state.get("follow_ups", assembled["follow_ups"]),
            "sources_used":    assembled["sources_used"],
            "chunks_used":     assembled["chunks_used"],
            "tokens_estimate": assembled["tokens_estimate"],
            "metadata": {
                "valid_citations":        validation["valid"],
                "hallucinated_citations": validation["hallucinated"],
                "uncited_sources":        validation["uncited"],
            },
        }
    except Exception as exc:
        logger.exception("[extract_citations] Failed: %s", exc)
        return {"error": str(exc), "failed_node": "extract_citations"}


def handle_error(state: dict) -> dict:
    logger.error(
        "[generation] FAILED at node='%s': %s",
        state.get("failed_node", "unknown"),
        state.get("error", "Unknown error"),
    )
    return {}