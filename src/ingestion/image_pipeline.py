"""
image_pipeline.py

LangGraph ingestion pipeline for image sources (PRD §2.5).

Stages:
  1. image_validate  — check format (PNG/JPG/WebP/GIF), get dimensions
  2. image_caption   — VLM via Ollama (LLaVA primary, moondream2 fallback)
  3. image_chunk     — parse VLM output + local OCR -> 3 child chunks
  4. image_embed     — embed + persist
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, Optional

from langchain_core.documents import Document
from langgraph.graph import END, StateGraph
from src.ingestion.state import IngestionState
from src.ingestion.nodes.utils import safe_node

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL  = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
VLM_PRIMARY      = os.getenv("VLM_PRIMARY",     "llava")
VLM_FALLBACK     = os.getenv("VLM_FALLBACK",    "moondream")

CAPTION_PROMPT = """Analyze this image and provide three distinct sections in your response:

[CAPTION]
Provide a detailed 1-2 paragraph description/caption of the main subject, context, and visual style of the image.

[OCR]
Transcribe every word, letter, number, and piece of visible text exactly as it appears in the image. If there is no text, write "No text present".

[REGIONS]
Identify the key regions, objects, or bounding areas in the image and describe what is present in each part (e.g. 'top-right: logo', 'center: bar chart').

Ensure you use the exact headers [CAPTION], [OCR], and [REGIONS] as delimiters in your response."""

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
    caption_content = state["caption"]
    file_path = state["file_path"]

    # Parse sections from VLM response
    caption_text = ""
    ocr_text = ""
    regions_text = ""

    try:
        caption_match = re.search(r"\[CAPTION\](.*?)(?=\[OCR\]|\[REGIONS\]|$)", caption_content, re.DOTALL | re.IGNORECASE)
        ocr_match = re.search(r"\[OCR\](.*?)(?=\[CAPTION\]|\[REGIONS\]|$)", caption_content, re.DOTALL | re.IGNORECASE)
        regions_match = re.search(r"\[REGIONS\](.*?)(?=\[CAPTION\]|\[OCR\]|$)", caption_content, re.DOTALL | re.IGNORECASE)

        if caption_match:
            caption_text = caption_match.group(1).strip()
        if ocr_match:
            ocr_text = ocr_match.group(1).strip()
        if regions_match:
            regions_text = regions_match.group(1).strip()
    except Exception as parse_err:
        logger.warning("[image_chunk] Failed to parse VLM output: %s", parse_err)

    if not caption_text:
        caption_text = caption_content.strip()
    if not ocr_text:
        ocr_text = "No text present"
    if not regions_text:
        regions_text = "General image content"

    # supplement VLM OCR using local pytesseract
    try:
        import pytesseract
        from PIL import Image
        local_ocr = pytesseract.image_to_string(Image.open(file_path)).strip()
        if local_ocr and len(local_ocr) > 3:
            logger.info("[image_chunk] Local pytesseract OCR text extracted (%d chars)", len(local_ocr))
            ocr_text = f"{ocr_text}\n\n[Local Tesseract OCR Extraction]:\n{local_ocr}"
    except Exception as ocr_err:
        logger.warning("[image_chunk] Local pytesseract OCR not available: %s", ocr_err)

    chunks = []
    
    # 1. Caption Child Chunk
    chunks.append(Document(
        page_content=caption_text,
        metadata={
            "source_id":         source_id,
            "source_type":       "image",
            "chunk_id":          f"{source_id}_caption",
            "chunk_index":       0,
            "child_type":        "caption",
            "strategy_used":     "vlm_caption",
            "image_path":        file_path,
            "image_dimensions":  state.get("image_dimensions", ""),
            "vlm_model_used":    state.get("vlm_model_used", ""),
        }
    ))

    # 2. OCR Child Chunk
    chunks.append(Document(
        page_content=ocr_text,
        metadata={
            "source_id":         source_id,
            "source_type":       "image",
            "chunk_id":          f"{source_id}_ocr",
            "chunk_index":       1,
            "child_type":        "ocr",
            "strategy_used":     "vlm_ocr",
            "image_path":        file_path,
            "image_dimensions":  state.get("image_dimensions", ""),
            "vlm_model_used":    state.get("vlm_model_used", ""),
        }
    ))

    # 3. Regions Child Chunk
    chunks.append(Document(
        page_content=regions_text,
        metadata={
            "source_id":         source_id,
            "source_type":       "image",
            "chunk_id":          f"{source_id}_regions",
            "chunk_index":       2,
            "child_type":        "region_description",
            "strategy_used":     "vlm_regions",
            "image_path":        file_path,
            "image_dimensions":  state.get("image_dimensions", ""),
            "vlm_model_used":    state.get("vlm_model_used", ""),
        }
    ))

    logger.info("[image_chunk] Created 3 child chunks: caption, ocr, region_description")
    return {"chunks": chunks}

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
        "[run_image_pipeline] Done — VLM description and OCR child chunks indexed."
    )
    return result
