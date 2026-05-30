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
from langchain.retrievers.document_compressors import LLMChainExtractor

logger = logging.getLogger(__name__)


class ContextualCompressor:
    """
    Extracts the relevant snippet from each document using an LLM.
    1 LLM call per document — use only when precision matters over cost.
    """

    def compress(
        self,
        query: str,
        docs:  List[Document],
    ) -> List[Document]:
        if not docs:
            return docs
        try:
            from src.generation.llm_registry import LLMRegistry
            llm        = LLMRegistry.get()
            compressor = LLMChainExtractor.from_llm(llm)
            compressed = compressor.compress_documents(docs, query)
            logger.info(
                "[ContextualCompressor] Compressed %d → %d docs",
                len(docs), len(compressed),
            )
            return compressed or docs   # fallback to originals if all filtered out
        except Exception as exc:
            logger.warning("[ContextualCompressor] Failed (%s) — returning original docs", exc)
            return docs
