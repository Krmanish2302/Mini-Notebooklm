"""
image_pipeline.py

LangGraph ingestion pipeline for image sources (PRD §2.5).

Stages:
  1. image_validate  — check format (PNG/JPG/WebP/GIF), get dimensions
  2. image_caption   — VLM via Ollama (LLaVA primary, moondream2 fallback)
  3. image_chunk     — single chunk (VLM output is already 100-250 words)
  4. image_embed     — embed + persist

Usage:
    from src.ingestion.image_pipeline import run_image_pipeline
    result = run_image_pipeline(file_path="diagram.png", source_id="img_001")
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from langchain_core.documents import Document
from langgraph.graph import END, StateGraph
from src.ingestion.state import IngestionState

from src.ingestion.nodes.utils import safe_node

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL  = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
VLM_PRIMARY      = os.getenv("VLM_PRIMARY",     "llava")
VLM_FALLBACK     = os.getenv("VLM_FALLBACK",    "moondream")

CAPTION_PROMPT = """Describe this image in comprehensive detail. Include:
1. Main subject(s) and what they are doing or showing
2. All visible text or numbers (transcribe exactly)
3. Charts, graphs, or data visualizations (describe all values, labels, trends)
4. Spatial relationships between elements
5. Colors, styles, or visual patterns that carry meaning
6. Any context clues about the purpose or source of this image

Be precise and factual. Do not speculate. Use technical language if appropriate."""


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
@safe_node("image_validate")
def image_validate(state: dict) -> dict:
    from PIL import Image

    file_path = state["file_path"]
    ALLOWED   = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
    ext = os.path.splitext(file_path)[-1].lower()
    if ext not in ALLOWED:
        raise ValueError(f"Unsupported image format: {ext}. Allowed: {ALLOWED}")

    with Image.open(file_path) as img:
        width, height = img.size
        fmt = img.format

    size_bytes = os.path.getsize(file_path)
    logger.info(
        "[image_validate] '%s' — %dx%d %s (%d bytes)",
        file_path, width, height, fmt, size_bytes,
    )
    return {
        "image_dimensions": f"{width}x{height}",
        "image_format":     fmt,
        "image_size_bytes": size_bytes,
    }


@safe_node("image_caption")
def image_caption(state: dict) -> dict:
    import base64

    from langchain_ollama import ChatOllama
    from langchain_core.messages import HumanMessage

    file_path = state["file_path"]

    with open(file_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    ext_to_mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                   ".png": "image/png",  ".webp": "image/webp",  ".gif": "image/gif"}
    ext  = os.path.splitext(file_path)[-1].lower()
    mime = ext_to_mime.get(ext, "image/png")

    vlm_model = VLM_PRIMARY
    caption   = None

    for model in [VLM_PRIMARY, VLM_FALLBACK]:
        try:
            llm = ChatOllama(model=model, base_url=OLLAMA_BASE_URL)
            msg = HumanMessage(content=[
                {"type": "text",       "text": CAPTION_PROMPT},
                {"type": "image_url",  "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
            ])
            response  = llm.invoke([msg])
            caption   = response.content
            vlm_model = model
            logger.info("[image_caption] Captioned with model=%s (%d chars)", model, len(caption))
            break
        except Exception as e:
            logger.warning("[image_caption] model=%s failed: %s", model, e)

    if not caption:
        raise RuntimeError("Both VLM models failed. Ensure Ollama is running.")

    return {"caption": caption, "vlm_model_used": vlm_model}


@safe_node("image_chunk")
def image_chunk(state: dict) -> dict:
    source_id = state["source_id"]
    caption   = state["caption"]
    file_path = state["file_path"]

    chunk = Document(
        page_content=caption,
        metadata={
            "source_id":         source_id,
            "source_type":       "image",
            "chunk_id":          f"{source_id}_0",
            "chunk_index":       0,
            "modality":          "image_caption",
            "image_path":        file_path,
            "image_dimensions":  state.get("image_dimensions", ""),
            "vlm_model_used":    state.get("vlm_model_used", ""),
            "caption_length":    len(caption.split()),
        },
    )
    logger.info("[image_chunk] Single caption chunk (%d tokens est)", chunk.metadata["caption_length"])
    return {"chunks": [chunk]}


@safe_node("image_embed")
def image_embed(state: dict) -> dict:
    from src.ingestion.nodes.embed_node import embed_and_index
    return embed_and_index(state)


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
def _build_image_graph() -> StateGraph:
    g = StateGraph(IngestionState)
    g.add_node("image_validate", image_validate)
    g.add_node("image_caption",  image_caption)
    g.add_node("image_chunk",    image_chunk)
    g.add_node("image_embed",    image_embed)

    g.set_entry_point("image_validate")
    g.add_edge("image_validate", "image_caption")
    g.add_edge("image_caption",  "image_chunk")
    g.add_edge("image_chunk",    "image_embed")
    g.add_edge("image_embed",    END)
    return g.compile()


image_app = _build_image_graph()


# ---------------------------------------------------------------------------
# Public runner
# ---------------------------------------------------------------------------
def run_image_pipeline(file_path: str, source_id: str, source_name: Optional[str] = None) -> Dict[str, Any]:
    init_state = {
        "file_path":   file_path,
        "source_id":   source_id,
        "source_type": "image",
        "source_name": source_name,
    }
    result = image_app.invoke(init_state)
    if result.get("error"):
        raise RuntimeError(f"Image pipeline failed: {result['error']}")
    logger.info(
        "[run_image_pipeline] Done — caption %d words, model=%s",
        result.get("caption", "").count(" ") + 1 if result.get("caption") else 0,
        result.get("vlm_model_used", "?"),
    )
    return result
