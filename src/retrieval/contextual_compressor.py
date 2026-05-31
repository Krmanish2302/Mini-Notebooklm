"""
contextual_compressor.py

Wraps LangChain LLMChainExtractor to extract only the relevant portions
of each retrieved document.

Use sparingly — makes 1 LLM call per document.
Set use_compression=False in RetrievalState to skip.

BUG-RET-02: was hardcoded ChatOpenAI/gpt-4o-mini — now uses LLMRegistry
so Groq/llama-3.1-70b is used consistently with the rest of the system.
"""
from __future__ import annotations
import logging
from typing import List

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class ContextualCompressor:
    """
    Local, API-free contextual compressor that filters document sentences
    based on local embedding similarity with the query.
    """

    def compress(
        self,
        query: str,
        docs:  List[Document],
    ) -> List[Document]:
        if not docs:
            return docs
        try:
            from src.ingestion.embedding.embedding_registry import EmbeddingRegistry
            import numpy as np
            import re

            embeddings_model = EmbeddingRegistry.get()
            q_emb = np.array(embeddings_model.embed_query(query))
            q_norm = np.linalg.norm(q_emb)

            if q_norm == 0:
                return docs

            compressed_docs = []
            for doc in docs:
                # Split page content into sentences
                sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", doc.page_content) if s.strip()]
                if not sentences:
                    continue

                # Embed sentences
                s_embeddings = embeddings_model.embed_documents(sentences)
                
                # Filter sentences by similarity threshold 0.40
                matching_sentences = []
                for sentence, s_emb in zip(sentences, s_embeddings):
                    s_emb = np.array(s_emb)
                    s_norm = np.linalg.norm(s_emb)
                    similarity = float(np.dot(q_emb, s_emb) / (q_norm * s_norm)) if s_norm > 0 else 0.0
                    
                    if similarity >= 0.40:
                        matching_sentences.append(sentence)

                if matching_sentences:
                    compressed_content = " ".join(matching_sentences)
                    compressed_docs.append(Document(
                        page_content=compressed_content,
                        metadata={**doc.metadata, "original_content": doc.page_content}
                    ))

            logger.info(
                "[ContextualCompressor] Compressed %d → %d docs locally",
                len(docs), len(compressed_docs),
            )
            return compressed_docs or docs   # fallback to originals if all filtered out
        except Exception as exc:
            logger.warning("[ContextualCompressor] Failed (%s) — returning original docs", exc)
            return docs
