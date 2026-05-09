"""
cross_modal_merger.py

Step 1 — Normalize:  Every pipeline output is converted to a unified
                     plain-text `content` field.  This is the
                     "convert everything to text first" step.

Step 2 — Merge:      Transcript and image-caption chunks are linked to
                     semantically related text chunks via cosine similarity
                     so the retriever can surface them together.

Pipeline position (from master_pipeline.py):
    raw_results  →  CrossModalMerger.normalize()  →  CrossModalMerger.merge()
    →  ContextualEnricher  →  AdaptiveChunker  →  EmbeddingPipeline
"""
import logging
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


class CrossModalMerger:
    """
    Normalizes multi-modal pipeline outputs to plain text and
    links related chunks across modalities.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.72,
        embedder=None,           # EmbeddingPipeline instance (optional for normalize-only use)
        max_related: int = 3,
    ):
        self.similarity_threshold = similarity_threshold
        self.embedder = embedder
        self.max_related = max_related

    # ── Step 1: Normalize ────────────────────────────────────────────────────

    def normalize(self, pipeline_result: Dict[str, Any], source_type: str) -> str:
        """
        Convert any pipeline output dict into a single clean text string.

        Every source type ultimately produces plain text — this step
        makes that conversion explicit and consistent.

        Args:
            pipeline_result: The dict returned by any *Pipeline.process().
            source_type:     One of: pdf | website | youtube | csv | image | text.

        Returns:
            A single clean plain-text string ready for chunking.
        """
        source_type = (source_type or "text").lower().strip()

        # PDF → join page texts with clear separators
        if source_type == "pdf":
            pages = pipeline_result.get("pages", [])
            if pages:
                return "\n\n".join(
                    f"[Page {p['page_number']}]\n{p['text'].strip()}"
                    for p in pages
                    if p.get("text", "").strip()
                )

        # YouTube / transcript → already plain text, clean timestamps
        if source_type in ("youtube", "transcript"):
            content = pipeline_result.get("content", "")
            return self._clean_transcript(content)

        # CSV → convert rows to readable prose lines
        if source_type in ("csv", "excel"):
            rows = pipeline_result.get("rows", [])
            if rows:
                lines = []
                for row in rows:
                    lines.append(
                        ", ".join(f"{k}: {v}" for k, v in row.items() if v is not None)
                    )
                return "\n".join(lines)

        # Image → use caption if present, else description
        if source_type == "image":
            caption = pipeline_result.get("caption", "")
            description = pipeline_result.get("description", "")
            return caption or description or pipeline_result.get("content", "")

        # Website / default → content field (already clean text from trafilatura)
        return pipeline_result.get("content", "")

    # ── Step 2: Link related chunks across modalities ────────────────────────

    def merge(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Add a `related_chunks` field to transcript and image chunks
        by finding semantically similar text chunks.

        If no embedder is configured, returns chunks unchanged
        (graceful degradation — merging is optional).

        Args:
            chunks: List of chunk dicts (output of AdaptiveChunker).

        Returns:
            Same list with `related_chunks` added where applicable.
        """
        if self.embedder is None:
            logger.debug("CrossModalMerger: no embedder — skipping similarity linking.")
            return chunks

        text_chunks = [
            c for c in chunks if c.get("modality") == "text"
        ]
        if not text_chunks:
            return chunks

        # Pre-compute text chunk embeddings once
        try:
            text_embs = self.embedder.embed(
                [c["content"] for c in text_chunks]
            )  # shape: (N, dim)
            text_norms = np.linalg.norm(text_embs, axis=1, keepdims=True)
            text_embs_normed = text_embs / (text_norms + 1e-9)
        except Exception as emb_err:
            logger.warning("CrossModalMerger: embedding failed — %s", emb_err)
            return chunks

        for chunk in chunks:
            if chunk.get("modality") not in ("image_caption", "transcript"):
                continue
            try:
                q_emb = self.embedder.embed_single(chunk["content"])
                q_norm = np.linalg.norm(q_emb)
                sims = text_embs_normed @ (q_emb / (q_norm + 1e-9))

                related = [
                    {"chunk_id": text_chunks[i]["id"], "similarity": float(sims[i])}
                    for i in range(len(text_chunks))
                    if sims[i] >= self.similarity_threshold
                ]
                related.sort(key=lambda x: x["similarity"], reverse=True)
                chunk["related_chunks"] = related[: self.max_related]
            except Exception as link_err:
                logger.warning(
                    "CrossModalMerger: could not link chunk %s — %s",
                    chunk.get("id"), link_err,
                )
                chunk["related_chunks"] = []

        return chunks

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _clean_transcript(text: str) -> str:
        """
        Remove YouTube timestamp markers like [00:42] from transcripts
        while preserving the spoken content.
        """
        import re
        # Remove [MM:SS] and [HH:MM:SS] markers
        cleaned = re.sub(r"\[\d{1,2}:\d{2}(?::\d{2})?\]\s*", "", text)
        # Collapse excessive blank lines
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()
