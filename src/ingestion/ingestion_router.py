"""
ingestion_router.py

Routes ingest requests to the correct source-type pipeline.

The user declares the source type explicitly (PRD §2.1) — no auto-detection:
  source_type in {"pdf", "youtube", "text", "image", "website"}

Public API:
    from src.ingestion.ingestion_router import ingest

    # PDF — run analyze first, then ingest with user choices
    stats  = ingest(source_type="pdf", file_path="doc.pdf", source_id="x", analyze_only=True)
    result = ingest(source_type="pdf", file_path="doc.pdf", source_id="x",
                    strategy="paragraph_based", embedding_dim=768)

    # Other types — single call
    result = ingest(source_type="youtube", file_path="https://youtu.be/...", source_id="y")
    result = ingest(source_type="text",    file_path="notes.md",              source_id="z")
    result = ingest(source_type="text",    content="Pasted text...",          source_id="p")
    result = ingest(source_type="image",   file_path="diagram.png",           source_id="i")
    result = ingest(source_type="website", file_path="https://example.com",   source_id="w")
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

SOURCE_TYPES = {"pdf", "youtube", "text", "image", "website"}


def ingest(
    source_type:   str,
    source_id:     str,
    file_path:     Optional[str] = None,
    content:       Optional[str] = None,   # text paste shortcut
    analyze_only:  bool = False,           # PDF only: return stats without chunking
    strategy:      str  = "paragraph_based",
    embedding_dim: int  = 384,
) -> Dict[str, Any]:
    """
    Route to the correct pipeline and run it.

    Returns a result dict with at minimum:
        num_chunks:      int
        vectorstore_path: str
        source_type:     str
    For PDF with analyze_only=True, returns the analysis stats dict.
    """
    if source_type not in SOURCE_TYPES:
        raise ValueError(f"Unknown source_type '{source_type}'. Must be one of {SOURCE_TYPES}")

    logger.info("[ingest] source_type=%s source_id=%s analyze_only=%s", source_type, source_id, analyze_only)

    # ── PDF ──────────────────────────────────────────────────────────────────
    if source_type == "pdf":
        if not file_path:
            raise ValueError("file_path required for PDF ingestion.")
        if analyze_only:
            from src.ingestion.pdf_pipeline import analyze_pdf
            return analyze_pdf(file_path=file_path, source_id=source_id)
        from src.ingestion.pdf_pipeline import run_pdf_pipeline
        return run_pdf_pipeline(
            file_path=file_path,
            source_id=source_id,
            strategy=strategy,
            embedding_dim=embedding_dim,
        )

    # ── YouTube ──────────────────────────────────────────────────────────────
    if source_type == "youtube":
        if not file_path:
            raise ValueError("file_path (URL) required for YouTube ingestion.")
        from src.ingestion.youtube_pipeline import run_youtube_pipeline
        return run_youtube_pipeline(url=file_path, source_id=source_id)

    # ── Text / Paste ─────────────────────────────────────────────────────────
    if source_type == "text":
        from src.ingestion.text_pipeline import run_text_pipeline
        return run_text_pipeline(
            source_id=source_id,
            file_path=file_path,
            content=content,
        )

    # ── Image ─────────────────────────────────────────────────────────────────
    if source_type == "image":
        if not file_path:
            raise ValueError("file_path required for image ingestion.")
        from src.ingestion.image_pipeline import run_image_pipeline
        return run_image_pipeline(file_path=file_path, source_id=source_id)

    # ── Website ───────────────────────────────────────────────────────────────
    if source_type == "website":
        if not file_path:
            raise ValueError("file_path (URL) required for website ingestion.")
        from src.ingestion.website_pipeline import run_website_pipeline
        return run_website_pipeline(url=file_path, source_id=source_id)
