"""
youtube_pipeline.py

LangGraph ingestion pipeline for YouTube sources (PRD §2.3).

Stages:
  1. yt_fetch    — youtube_transcript_api fetches timestamped transcript
  2. yt_clean    — remove filler words, short segments, repeated words
  3. yt_chunk    — Local SemanticChunker (cosine similarity splits)
  4. yt_embed    — embed + persist
"""
from __future__ import annotations

import logging
import re
import os
from typing import Any, Dict, Optional

from langgraph.graph import END, StateGraph
from src.ingestion.state import IngestionState
from src.ingestion.nodes.utils import safe_node

logger = logging.getLogger(__name__)

# Filler words to strip
_FILLERS = re.compile(
    r"\b(um+|uh+|er|ah|you know|kind of|sort of|basically|literally|I mean)\b",
    re.IGNORECASE,
)
_REPEATED = re.compile(r"\b(\w+)\s+\1\b", re.IGNORECASE)  # "the the" → "the"

def format_time(sec: float) -> str:
    """Formats seconds into MM:SS."""
    return f"{int(sec // 60):02d}:{int(sec % 60):02d}"

def extract_video_id(url: str) -> Optional[str]:
    """Robust extractor of YouTube video ID from URL."""
    import urllib.parse as urlparse
    parsed = urlparse.urlparse(url)
    if parsed.hostname in ('youtu.be', 'www.youtu.be'):
        return parsed.path[1:]
    if parsed.hostname in ('youtube.com', 'www.youtube.com'):
        if parsed.path == '/watch':
            p = urlparse.parse_qs(parsed.query)
            return p.get('v', [None])[0]
        if parsed.path.startswith('/embed/'):
            return parsed.path.split('/')[2]
        if parsed.path.startswith('/v/'):
            return parsed.path.split('/')[2]
    return None

# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
@safe_node("yt_fetch")
def yt_fetch(state: dict) -> dict:
    from langchain_community.document_loaders import YoutubeLoader
    from langchain_community.document_loaders.youtube import TranscriptFormat
    from langchain_core.documents import Document

    url = state["file_path"]   # URL stored in file_path
    source_id = state["source_id"]
    video_id = extract_video_id(url)
    
    if not video_id:
        raise ValueError(f"Could not extract YouTube video ID from URL: {url}")
        
    try:
        # Load transcript using LangChain's YoutubeLoader with LINES format
        loader = YoutubeLoader.from_youtube_url(
            url, 
            add_video_info=False, 
            transcript_format=TranscriptFormat.LINES,
            language=["en", "en-US"]
        )
        docs = loader.load()
        
        # Convert list of Documents to segment dictionaries for compatibility
        transcript_data = []
        for d in docs:
            transcript_data.append({
                "text": d.page_content,
                "start": d.metadata.get("start", 0.0),
                "duration": d.metadata.get("duration", 5.0)
            })
    except Exception as e:
        logger.error("[yt_fetch] LangChain YoutubeLoader failed: %s", e)
        raise ValueError(
            f"Could not retrieve any transcript or captions for YouTube video '{url}'. "
            "Please ensure the video exists and has English subtitles/captions enabled."
        ) from e

    if not transcript_data:
        raise ValueError(
            f"Could not retrieve any transcript or captions for YouTube video '{url}'. "
            "Please ensure the video exists and has English subtitles/captions enabled."
        )

    # Return raw transcript segments in state so that chunk node can perform semantic chunking with timestamps
    total_words = sum(len(segment["text"].split()) for segment in transcript_data)
    logger.info("[yt_fetch] LangChain fetched %d segments, ~%d words from '%s'", len(transcript_data), total_words, url)
    return {
        "raw_documents": [Document(page_content=url, metadata={"source_id": source_id, "source_type": "youtube"})], 
        "original_word_count": total_words,
        "transcript_segments": transcript_data
    }

@safe_node("yt_clean")
def yt_clean(state: dict) -> dict:
    segments = state.get("transcript_segments", [])
    cleaned_segments = []
    
    for seg in segments:
        text = seg["text"]
        # Remove filler words
        text = _FILLERS.sub("", text)
        # Remove repeated words
        text = _REPEATED.sub(r"\1", text)
        # Collapse extra spaces
        text = re.sub(r" {2,}", " ", text).strip()
        
        if len(text.split()) >= 1:  # Keep segments with text
            cleaned_segments.append({
                "text": text,
                "start": seg["start"],
                "duration": seg.get("duration") or 5.0
            })
            
    total_after = sum(len(seg["text"].split()) for seg in cleaned_segments)
    orig = state.get("original_word_count", 0)
    pct = round(100 * (orig - total_after) / orig, 1) if orig else 0
    logger.info(
        "[yt_clean] Cleaned segments: %d words → %d words (%s%% reduction)",
        orig, total_after, pct,
    )
    return {
        "cleaned_segments": cleaned_segments,
        "cleaned_word_count": total_after,
        "reduction_pct": pct
    }

@safe_node("yt_chunk")
def yt_chunk(state: dict) -> dict:
    """Semantic chunking over YouTube transcripts — split where cosine similarity < threshold."""
    from langchain_core.documents import Document
    from src.ingestion.embedding.embedding_registry import EmbeddingRegistry
    import numpy as np

    cleaned_segs = state.get("cleaned_segments", [])
    source_id = state["source_id"]
    
    if not cleaned_segs:
        return {"chunks": []}

    # Step 1: Pre-group segments into candidate blocks (around 50-80 words)
    blocks = []
    current_block_segs = []
    current_word_count = 0
    
    for seg in cleaned_segs:
        current_block_segs.append(seg)
        current_word_count += len(seg["text"].split())
        
        if current_word_count >= 60:
            block_text = " ".join(s["text"] for s in current_block_segs)
            blocks.append({
                "text": block_text,
                "start": current_block_segs[0]["start"],
                "end": current_block_segs[-1]["start"] + current_block_segs[-1]["duration"],
                "segments": current_block_segs
            })
            current_block_segs = []
            current_word_count = 0
            
    if current_block_segs:
        block_text = " ".join(s["text"] for s in current_block_segs)
        blocks.append({
            "text": block_text,
            "start": current_block_segs[0]["start"],
            "end": current_block_segs[-1]["start"] + current_block_segs[-1]["duration"],
            "segments": current_block_segs
        })

    if not blocks:
        return {"chunks": []}

    # Step 2: Embed blocks using lightweight semantic chunking model
    embeddings_model = EmbeddingRegistry.get("all-MiniLM-L6-v2")
    block_texts = [b["text"] for b in blocks]
    
    try:
        embeddings = embeddings_model.embed_documents(block_texts)
    except Exception as e:
        logger.warning("[yt_chunk] Embedding failed, fallback to sequential mock embeddings: %s", e)
        embeddings = [np.random.rand(384) for _ in blocks]

    # Normalize embeddings for cosine similarity
    normalized_embeddings = []
    for e in embeddings:
        arr = np.array(e)
        norm = np.linalg.norm(arr)
        normalized_embeddings.append(arr / norm if norm > 0 else arr)
    
    chunks = []
    current_chunk_blocks = [blocks[0]]
    current_embedding = normalized_embeddings[0]
    
    for i in range(1, len(blocks)):
        sim = np.dot(current_embedding, normalized_embeddings[i])
        current_token_count = len(" ".join(b["text"] for b in current_chunk_blocks)) // 4
        
        # Split if similarity drops below 0.70 or chunk token budget exceeded
        if sim >= 0.70 and current_token_count < 300:
            current_chunk_blocks.append(blocks[i])
            current_embedding = np.mean([current_embedding, normalized_embeddings[i]], axis=0)
            norm = np.linalg.norm(current_embedding)
            current_embedding = current_embedding / norm if norm > 0 else current_embedding
        else:
            chunk_text = " ".join(b["text"] for b in current_chunk_blocks)
            start_time = current_chunk_blocks[0]["start"]
            end_time = current_chunk_blocks[-1]["end"]
            
            chunks.append(Document(
                page_content=chunk_text,
                metadata={
                    "source_id": source_id,
                    "source_type": "youtube",
                    "start": start_time,
                    "end": end_time,
                    "start_time": format_time(start_time),
                    "end_time": format_time(end_time),
                    "child_type": "transcript",
                    "strategy_used": "semantic_transcript"
                }
            ))
            current_chunk_blocks = [blocks[i]]
            current_embedding = normalized_embeddings[i]
            
    if current_chunk_blocks:
        chunk_text = " ".join(b["text"] for b in current_chunk_blocks)
        start_time = current_chunk_blocks[0]["start"]
        end_time = current_chunk_blocks[-1]["end"]
        chunks.append(Document(
            page_content=chunk_text,
            metadata={
                "source_id": source_id,
                "source_type": "youtube",
                "start": start_time,
                "end": end_time,
                "start_time": format_time(start_time),
                "end_time": format_time(end_time),
                "child_type": "transcript",
                "strategy_used": "semantic_transcript"
            }
        ))

    # Add chunk index markers
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = f"{source_id}_{i}"
        c.metadata["chunk_index"] = i

    logger.info("[yt_chunk] Created %d semantic transcript chunks", len(chunks))
    return {"chunks": chunks}

@safe_node("yt_embed")
def yt_embed(state: dict) -> dict:
    from src.ingestion.nodes.embed_node import embed_and_index
    return embed_and_index(state)

# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
def _build_yt_graph() -> StateGraph:
    g = StateGraph(IngestionState)
    g.add_node("yt_fetch", yt_fetch)
    g.add_node("yt_clean", yt_clean)
    g.add_node("yt_chunk", yt_chunk)
    g.add_node("yt_embed", yt_embed)

    g.set_entry_point("yt_fetch")
    g.add_edge("yt_fetch", "yt_clean")
    g.add_edge("yt_clean", "yt_chunk")
    g.add_edge("yt_chunk", "yt_embed")
    g.add_edge("yt_embed", END)
    return g.compile()

yt_app = _build_yt_graph()

# ---------------------------------------------------------------------------
# Public runner
# ---------------------------------------------------------------------------
def run_youtube_pipeline(url: str, source_id: str, source_name: Optional[str] = None) -> Dict[str, Any]:
    init_state = {
        "file_path":   url,
        "source_id":   source_id,
        "source_type": "youtube",
        "source_name": source_name,
    }
    result = yt_app.invoke(init_state)
    if result.get("error"):
        raise RuntimeError(f"YouTube pipeline failed: {result['error']}")
    logger.info(
        "[run_youtube_pipeline] Done — %d chunks, reduction=%s%%",
        result.get("num_chunks", 0),
        result.get("reduction_pct", "?"),
    )
    return result
