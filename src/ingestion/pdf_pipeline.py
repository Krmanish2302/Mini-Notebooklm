"""
pdf_pipeline.py

LangGraph ingestion pipeline for PDF sources.

Stages (PRD §2.2):
  1. extract_node   — PyMuPDF page-by-page extraction
  2. analyze_node   — 20% sample → stats for every chunking strategy
  3. [PAUSE]        — caller presents stats, passes user choice back into state
  4. chunk_node     — chunk with chosen strategy
  5. embed_node     — embed + persist to MultiFAISSStore

Usage:
    from src.ingestion.pdf_pipeline import run_pdf_pipeline, analyze_pdf

    # Step 1 — get analysis stats (for UI to display)
    stats = analyze_pdf(file_path, source_id)

    # Step 2 — user picks strategy + model, then run full pipeline
    result = run_pdf_pipeline(
        file_path=file_path,
        source_id=source_id,
        strategy="paragraph_based",   # user choice
        embedding_dim=384,            # user choice
    )
"""
from __future__ import annotations

import logging
import os
import re
import statistics
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langgraph.graph import END, StateGraph
from src.ingestion.state import IngestionState

from src.ingestion.nodes.utils import safe_node

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VECTOR_STORE_DIR = os.getenv("VECTOR_STORE_DIR", "data/vectorstores")
ANALYSIS_SAMPLE  = float(os.getenv("PDF_ANALYSIS_SAMPLE", "0.20"))  # 20%

STRATEGY_DESCRIPTIONS = {
    "page_based":       "One chunk per page",
    "paragraph_based":  "Split at blank lines",
    "sentence_based":   "Each sentence is a chunk",
    "fixed_256":        "Fixed 256-token windows",
    "fixed_512":        "Fixed 512-token windows",
    "chapter_based":    "Split at detected headings",
    "semantic":         "Split where semantic similarity drops",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _token_estimate(text: str) -> int:
    return int(len(text.split()) * 1.33)


def _detect_headings(text: str) -> bool:
    patterns = [
        r"^(Chapter|CHAPTER|Section|SECTION|Part|PART)\s+\d+",
        r"^[A-Z][A-Z\s]{4,}$",
    ]
    for line in text.split("\n"):
        for p in patterns:
            if re.match(p, line.strip()):
                return True
    return False


def _chunks_for_strategy(docs: List[Document], strategy: str) -> List[str]:
    """Return list of chunk texts for a strategy — used only for stat computation."""
    texts = [d.page_content for d in docs]

    if strategy == "page_based":
        return texts

    if strategy == "paragraph_based":
        result = []
        for t in texts:
            result.extend([p.strip() for p in t.split("\n\n") if p.strip()])
        return result

    if strategy == "sentence_based":
        result = []
        for t in texts:
            result.extend([s.strip() for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()])
        return result

    if strategy == "fixed_256":
        result = []
        for t in texts:
            words = t.split()
            for i in range(0, len(words), 256):
                chunk = " ".join(words[i:i+256])
                if chunk:
                    result.append(chunk)
        return result

    if strategy == "fixed_512":
        result = []
        for t in texts:
            words = t.split()
            for i in range(0, len(words), 512):
                chunk = " ".join(words[i:i+512])
                if chunk:
                    result.append(chunk)
        return result

    if strategy == "chapter_based":
        heading_re = re.compile(
            r"^(Chapter|CHAPTER|Section|SECTION|Part|PART)\s+\d+|^[A-Z][A-Z\s]{4,}$",
            re.MULTILINE,
        )
        full = "\n\n".join(texts)
        parts = heading_re.split(full)
        return [p.strip() for p in parts if p and len(p.split()) > 10]

    # semantic — approximate with paragraph_based for stat purposes
    result = []
    for t in texts:
        result.extend([p.strip() for p in t.split("\n\n") if p.strip()])
    return result


def _compute_stats(chunks: List[str]) -> Dict[str, Any]:
    if not chunks:
        return {}
    token_counts = [_token_estimate(c) for c in chunks]
    mean   = statistics.mean(token_counts)
    median = statistics.median(token_counts)
    stddev = statistics.stdev(token_counts) if len(token_counts) > 1 else 0.0
    under50  = sum(1 for t in token_counts if t < 50)
    over400  = sum(1 for t in token_counts if t > 400)
    return {
        "estimated_chunks": len(chunks),
        "avg_tokens":        round(mean, 1),
        "median_tokens":     round(median, 1),
        "std_dev":           round(stddev, 1),
        "min_tokens":        min(token_counts),
        "max_tokens":        max(token_counts),
        "pct_under_50":      round(100 * under50 / len(chunks), 1),
        "pct_over_400":      round(100 * over400  / len(chunks), 1),
    }


def _recommend(stats_by_strategy: Dict[str, Dict], has_headings: bool) -> str:
    if has_headings:
        return "chapter_based"
    para_avg = stats_by_strategy.get("paragraph_based", {}).get("avg_tokens", 150)
    if para_avg > 300:
        return "fixed_512"
    if para_avg < 80:
        return "sentence_based"
    return "paragraph_based"


def _recommend_embedding(avg_tokens: float) -> str:
    if avg_tokens < 200:
        return "all-MiniLM-L6-v2"
    return "nomic-embed-text-v1.5"


# ---------------------------------------------------------------------------
# Public: analyze_pdf (call before showing UI)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Public: analyze_pdf (call before showing UI)
# ---------------------------------------------------------------------------
def chunk_text_fixed(text: str, chunk_size: int = 500) -> List[str]:
    chunks = []
    current_chunk = ''
    words = text.split()
    for word in words:
        if len(current_chunk) + len(word) + 1 <= chunk_size:
            current_chunk += (word + ' ')
        else:
            chunks.append(current_chunk.strip())
            current_chunk = word + ' '
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


def simulate_fixed_chunking(pages_metrics: List[Dict[str, Any]], chunk_size: int = 500) -> int:
    total_chunks = 0
    for page in pages_metrics:
        chunks = chunk_text_fixed(page["text"], chunk_size=chunk_size)
        total_chunks += len(chunks)
    return total_chunks


def simulate_semantic_chunking(pages_metrics: List[Dict[str, Any]], similarity_threshold: float = 0.75, max_tokens: int = 500) -> int:
    try:
        import nltk
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        from src.ingestion.embedding.embedding_registry import EmbeddingRegistry
        embeddings_model = EmbeddingRegistry.get()
    except Exception as e:
        logger.warning("Failed to initialize semantic chunking models for simulation: %s", e)
        total_sentences = sum(page["page_sentence_count"] for page in pages_metrics)
        return max(1, int(total_sentences / 4))

    total_chunks = 0
    for page in pages_metrics:
        sentences = nltk.sent_tokenize(page["text"])
        if not sentences:
            continue
        try:
            embeddings = embeddings_model.embed_documents(sentences)
            if not embeddings or len(embeddings) == 0:
                continue
            
            chunks = []
            current_chunk = [sentences[0]]
            current_embedding = embeddings[0]

            for i in range(1, len(sentences)):
                sim = np.dot(current_embedding, embeddings[i])
                chunk_token_count = len(" ".join(current_chunk)) // 4

                if sim >= similarity_threshold and chunk_token_count < max_tokens:
                    current_chunk.append(sentences[i])
                    current_embedding = np.mean([current_embedding, embeddings[i]], axis=0)
                else:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [sentences[i]]
                    current_embedding = embeddings[i]

            if current_chunk:
                chunks.append(" ".join(current_chunk))
            total_chunks += len(chunks)
        except Exception as exc:
            logger.warning("Failed during semantic chunk simulation for page: %s", exc)
            total_chunks += max(1, int(len(sentences) / 4))
            
    return total_chunks


def simulate_recursive_chunking(pages_metrics: List[Dict[str, Any]], chunk_size: int = 800, chunk_overlap: int = 100) -> int:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    total_chunks = 0
    for page in pages_metrics:
        doc = Document(page_content=page["text"])
        chunks = splitter.split_documents([doc])
        total_chunks += len(chunks)
    return total_chunks
def analyze_pdf(file_path: str, source_id: str, start_page: int = 1) -> Dict[str, Any]:
    """
    Analyzes all active pages starting from start_page for metrics,
    and runs strategy simulations on a 10-page slice.
    Returns detailed metrics and strategy comparisons extrapolated.
    """
    from langchain_community.document_loaders import PyMuPDFLoader

    docs: List[Document] = PyMuPDFLoader(file_path).load()
    total_pages = len(docs)
    
    start_idx = min(total_pages - 1, max(0, start_page - 1))
    
    # 1. Parse metrics for all pages (starting from start_page to the end)
    pages_metrics = []
    active_docs = docs[start_idx:]
    for i, doc in enumerate(active_docs):
        text = doc.page_content or ""
        import re
        raw_paras = re.split(r'\n\s*\n', text)
        cleaned_paras = []
        for p in raw_paras:
            p_clean = re.sub(r'\s*\n\s*', ' ', p).strip()
            p_clean = re.sub(r' +', ' ', p_clean)
            if p_clean:
                cleaned_paras.append(p_clean)
        
        cleaned_text_with_paras = "\n\n".join(cleaned_paras)
        
        try:
            import nltk
            sentences = nltk.sent_tokenize(cleaned_text_with_paras)
            sentence_count = len(sentences)
        except Exception:
            sentence_count = len(cleaned_text_with_paras.split(". "))

        pages_metrics.append({
            "page_number": start_idx + i + 1,
            "page_char_count": len(cleaned_text_with_paras),
            "page_word_count": len(cleaned_text_with_paras.split()),
            "page_sentence_count": sentence_count,
            "page_token_count": int(len(cleaned_text_with_paras) / 4),
            "text": cleaned_text_with_paras
        })

    # Run simulations on up to a 10-page slice for speed
    slice_metrics = pages_metrics[:10]
    analyzed_pages = len(slice_metrics)
    
    fixed_chunks_sim = simulate_fixed_chunking(slice_metrics, chunk_size=500)
    semantic_chunks_sim = simulate_semantic_chunking(slice_metrics, similarity_threshold=0.75, max_tokens=500)
    recursive_chunks_sim = simulate_recursive_chunking(slice_metrics, chunk_size=800, chunk_overlap=100)

    # Extrapolate to the active page range
    active_pages = total_pages - start_idx
    extrapolation_ratio = active_pages / max(1, analyzed_pages)

    estimated_fixed = int(fixed_chunks_sim * extrapolation_ratio)
    estimated_semantic = int(semantic_chunks_sim * extrapolation_ratio)
    estimated_recursive = int(recursive_chunks_sim * extrapolation_ratio)

    total_words_est = sum(p["page_word_count"] for p in pages_metrics)
    total_words_extrapolated = total_words_est
    return {
        "source_id":             source_id,
        "file_path":             file_path,
        "total_pages":           total_pages,
        "start_page":            start_page,
        "active_pages":          active_pages,
        "analyzed_pages":        analyzed_pages,
        "total_words_estimated": total_words_extrapolated,
        "pages_metrics":         pages_metrics,
        "strategies": {
            "fixed_size": {
                "estimated_chunks": estimated_fixed,
                "label": "Fixed-Size (500 chars)",
                "description": "Splits text at fixed character counts"
            },
            "semantic": {
                "estimated_chunks": estimated_semantic,
                "label": "Semantic Similarity (threshold 0.75)",
                "description": "Splits text based on sentence similarity using local embeddings"
            },
            "recursive": {
                "estimated_chunks": estimated_recursive,
                "label": "Recursive Character (800 chars)",
                "description": "Splits using hierarchical separators (paragraphs, sentences)"
            }
        }
    }


# ---------------------------------------------------------------------------
# LangGraph nodes
# ---------------------------------------------------------------------------
@safe_node("pdf_extract")
def pdf_extract(state: dict) -> dict:
    from langchain_community.document_loaders import PyMuPDFLoader
    file_path = state["file_path"]
    docs = PyMuPDFLoader(file_path).load()
    total_pages = len(docs)
    
    start_page = int(state.get("start_page", 1))
    start_idx = min(total_pages - 1, max(0, start_page - 1))
    docs = docs[start_idx:]
    
    # tag each page
    for i, doc in enumerate(docs):
        doc.metadata.update({
            "source_id":   state["source_id"],
            "source_type": "pdf",
            "page_number": start_idx + i + 1,
        })
    logger.info("[pdf_extract] %d pages loaded from '%s' starting from page %d", len(docs), file_path, start_page)
    return {"raw_documents": docs, "total_pages": total_pages}


@safe_node("pdf_chunk")
def pdf_chunk(state: dict) -> dict:
    """Chunk with strategy chosen by user (default: paragraph_based)."""
    docs      = state.get("raw_documents", [])
    strategy  = state.get("strategy", "paragraph_based")
    source_id = state["source_id"]

    if strategy == "fixed_size":
        from langchain_core.documents import Document
        chunks = []
        for doc in docs:
            text = doc.page_content or ""
            text_chunks = chunk_text_fixed(text, chunk_size=500)
            for i, chunk_text in enumerate(text_chunks):
                chunks.append(Document(
                    page_content=chunk_text,
                    metadata={**doc.metadata}
                ))

    elif strategy == "semantic":
        from langchain_core.documents import Document
        import nltk
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        from src.ingestion.embedding.embedding_registry import EmbeddingRegistry

        embeddings_model = EmbeddingRegistry.get()
        chunks = []
        for doc in docs:
            text = doc.page_content or ""
            sentences = nltk.sent_tokenize(text)
            if not sentences:
                continue
            try:
                embeddings = embeddings_model.embed_documents(sentences)
                if not embeddings or len(embeddings) == 0:
                    continue
                
                current_chunk = [sentences[0]]
                current_embedding = embeddings[0]

                for i in range(1, len(sentences)):
                    sim = np.dot(current_embedding, embeddings[i])
                    chunk_token_count = len(" ".join(current_chunk)) // 4

                    if sim >= 0.75 and chunk_token_count < 500:
                        current_chunk.append(sentences[i])
                        current_embedding = np.mean([current_embedding, embeddings[i]], axis=0)
                    else:
                        chunks.append(Document(
                            page_content=" ".join(current_chunk),
                            metadata={**doc.metadata}
                        ))
                        current_chunk = [sentences[i]]
                        current_embedding = embeddings[i]

                if current_chunk:
                    chunks.append(Document(
                        page_content=" ".join(current_chunk),
                        metadata={**doc.metadata}
                    ))
            except Exception as exc:
                logger.warning("Semantic chunking failed: %s, falling back", exc)
                from langchain_text_splitters import RecursiveCharacterTextSplitter
                splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
                chunks.extend(splitter.split_documents([doc]))

    elif strategy == "paragraph_based" or strategy == "paragraph":
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, chunk_overlap=80,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(docs)

    elif strategy == "fixed_512":
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

    elif strategy == "fixed_256":
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(docs)

    elif strategy == "sentence_based" or strategy == "sentence":
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300, chunk_overlap=30,
            separators=[". ", "! ", "? ", "\n", " ", ""],
        )
        chunks = splitter.split_documents(docs)

    elif strategy == "chapter_based" or strategy == "chapter":
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000, chunk_overlap=200,
            separators=["\n\n\n", "\n\n", "\n", ". ", " "],
        )
        chunks = splitter.split_documents(docs)

    elif strategy == "page_based" or strategy == "page":
        chunks = docs

    else:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, chunk_overlap=80,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(docs)

    for i, c in enumerate(chunks):
        c.metadata["chunk_id"]       = f"{source_id}_{i}"
        c.metadata["chunk_index"]    = i
        c.metadata["strategy_used"]  = strategy
        c.metadata["source_type"]    = "pdf"

    logger.info("[pdf_chunk] strategy=%s → %d chunks", strategy, len(chunks))
    return {"chunks": chunks}


@safe_node("pdf_embed")
def pdf_embed(state: dict) -> dict:
    from src.ingestion.nodes.embed_node import embed_and_index
    return embed_and_index(state)


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------
def _build_pdf_graph() -> StateGraph:
    g = StateGraph(IngestionState)
    g.add_node("pdf_extract", pdf_extract)
    g.add_node("pdf_chunk",   pdf_chunk)
    g.add_node("pdf_embed",   pdf_embed)

    g.set_entry_point("pdf_extract")
    g.add_edge("pdf_extract", "pdf_chunk")
    g.add_edge("pdf_chunk",   "pdf_embed")
    g.add_edge("pdf_embed",   END)
    return g.compile()


pdf_app = _build_pdf_graph()


# ---------------------------------------------------------------------------
# Public runner
# ---------------------------------------------------------------------------
def run_pdf_pipeline(
    file_path:     str,
    source_id:     str,
    strategy:      str = "paragraph_based",
    embedding_dim: int = 384,
    source_name:   Optional[str] = None,
    start_page:    int = 1,
    embedding_model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the full PDF ingestion pipeline.
    Call analyze_pdf() first to get stats, then pass user choices here.
    """
    init_state = {
        "file_path":      file_path,
        "source_id":      source_id,
        "strategy":       strategy,
        "embedding_dim":  embedding_dim,
        "source_type":    "pdf",
        "source_name":    source_name,
        "start_page":     start_page,
        "embedding_model": embedding_model,
    }
    result = pdf_app.invoke(init_state)
    if result.get("error"):
        raise RuntimeError(f"PDF pipeline failed: {result['error']}")
    logger.info(
        "[run_pdf_pipeline] Done — %d chunks, store='%s'",
        result.get("num_chunks", 0),
        result.get("vectorstore_path", ""),
    )
    return result
