"""
preprocess_node.py

LangGraph node: clean and normalise raw Document objects.

Operations applied to every document's page_content:
  1. Strip excessive whitespace / line breaks
  2. Remove PDF header/footer noise patterns  (page numbers, watermarks)
  3. Fix broken hyphenated words across line breaks  (e.g. "infor-\nmation")
  4. Filter out near-empty documents (< MIN_WORDS words)
  5. Propagate source metadata to all cleaned docs

This node intentionally stays lightweight and deterministic — no LLM calls.
"""
from __future__ import annotations

import re
import logging
from typing import List
from langchain_core.documents import Document
from .utils import safe_node

logger = logging.getLogger(__name__)

MIN_WORDS = 10   # discard pages with fewer words than this


def _clean_text(text: str) -> str:
    """Apply deterministic cleaning rules to raw page text."""
    # Fix hyphenated line-breaks: "infor-\nmation" → "information"
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    # Collapse 3+ newlines to double newline (paragraph boundary)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces / tabs to single space
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Remove standalone page numbers  e.g. "\n14\n" or "Page 14"
    text = re.sub(r"(?i)\bpage\s*\d+\b", "", text)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    return text.strip()


@safe_node("preprocess")
def preprocess(state: dict) -> dict:
    """
    LangGraph node — clean raw documents.

    Reads:  state["raw_documents"]
    Writes: state["cleaned_documents"]
    """
    raw_docs: List[Document] = state.get("raw_documents", [])
    cleaned: List[Document] = []

    for doc in raw_docs:
        clean_content = _clean_text(doc.page_content)
        word_count    = len(clean_content.split())

        if word_count < MIN_WORDS:
            logger.debug(
                "[preprocess] Skipping near-empty page (source_id=%s, words=%d)",
                doc.metadata.get("source_id", ""), word_count,
            )
            continue

        cleaned.append(
            Document(
                page_content=clean_content,
                metadata={**doc.metadata, "word_count": word_count},
            )
        )

    logger.info(
        "[preprocess] %d → %d documents after cleaning",
        len(raw_docs), len(cleaned),
    )
    return {"cleaned_documents": cleaned}
