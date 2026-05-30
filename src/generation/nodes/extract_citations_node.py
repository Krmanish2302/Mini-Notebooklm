"""
extract_citations_node.py — LangGraph node: extract citations + final assembly.

Fixes applied
-------------
* FIX #9: follow_ups precedence — use `or` instead of dict default so that
          ResponseGenerator's extraction is used when ResponseParser returns [].
"""
from __future__ import annotations
import logging

logger = logging.getLogger(__name__)


def extract_citations(state: dict) -> dict:
    try:
        answer    = state.get("answer", "")
        documents = state.get("documents", [])

        from src.generation.citation_extractor import CitationExtractor
        from src.generation.response_generator import ResponseGenerator

        chunks = []
        for i, doc in enumerate(documents, 1):
            if hasattr(doc, "page_content"):
                content = doc.page_content
                meta    = doc.metadata or {}
            else:
                content = doc.get("content", "")
                meta    = {k: v for k, v in doc.items() if k != "content"}
            chunks.append({"citation_label": f"S{i}", "content": content, **meta})

        extractor  = CitationExtractor(chunks)
        citations  = extractor.extract(answer)
        validation = extractor.validate(answer)

        generator = ResponseGenerator(chunks)
        assembled = generator.assemble(
            raw_llm_output=state.get("raw_output", answer),
            query=state.get("query", ""),
        )

        return {
            "answer":          assembled["answer"],
            "citations":       citations,
            # FIX #9: use `or` so ResponseGenerator's follow_ups are used when
            # ResponseParser returned an empty list (not just when key is absent)
            "follow_ups":      state.get("follow_ups") or assembled["follow_ups"],
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
    # FIX #11: return safe defaults so callers never get KeyError on answer/citations
    return {
        "answer":          "",
        "citations":       [],
        "follow_ups":      [],
        "sources_used":    [],
        "chunks_used":     [],
        "tokens_estimate": 0,
    }
