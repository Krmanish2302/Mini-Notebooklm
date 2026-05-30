"""
contextual_compressor.py

Wraps LangChain ContextualCompressionRetriever with LLMChainExtractor.
Extracts only the relevant portions of each retrieved document.

Use sparingly — makes 1 LLM call per document.
Set use_compression=False in RetrievalState to skip.
"""
from __future__ import annotations
import logging
import os
from typing import List

from langchain_core.documents import Document
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever

logger = logging.getLogger(__name__)
COMPRESSION_MODEL = os.getenv("COMPRESSION_MODEL", "gpt-4o-mini")


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
            from langchain_openai import ChatOpenAI
            llm        = ChatOpenAI(model=COMPRESSION_MODEL, temperature=0)
            compressor = LLMChainExtractor.from_llm(llm)
            compressed = compressor.compress_documents(docs, query)
            logger.info(
                "[ContextualCompressor] Compressed %d → %d docs",
                len(docs), len(compressed),
            )
            return compressed or docs   # fallback to originals if compression removes everything
        except Exception as exc:
            logger.warning("[ContextualCompressor] Failed (%s) — returning original docs", exc)
            return docs