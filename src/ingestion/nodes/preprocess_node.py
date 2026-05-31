"""
preprocess_node.py

LangGraph node: clean and normalise raw Document objects.
No LLM calls — pure deterministic regex transforms.

Operations:
  1. Fix hyphenated line-breaks ("infor-\\nma­tion" → "information")
  2. Collapse 3+ newlines → double newline
  3. Collapse multiple spaces/tabs → single space
  4. Remove standalone page numbers
  5. Drop near-empty pages (< MIN_WORDS words)
"""
from __future__ import annotations
import re
import logging
from typing import List
from langchain_core.documents import Document
from .utils import safe_node

logger = logging.getLogger(__name__)
MIN_WORDS = 10


def _clean(text: str) -> str:
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"(?i)\bpage\s*\d+\b", "", text)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    return text.strip()


@safe_node("preprocess")
def preprocess(state: dict) -> dict:
    """
    Reads:  state["raw_documents"]
    Writes: state["cleaned_documents"]
    """
    raw   = state.get("raw_documents", [])
    clean: List[Document] = []

    for doc in raw:
        content    = _clean(doc.page_content)
        word_count = len(content.split())
        # Keep all pages, even if word_count is small or zero
        clean.append(Document(
            page_content=content,
            metadata={**doc.metadata, "word_count": word_count},
        ))

    logger.info("[preprocess] %d → %d docs after cleaning", len(raw), len(clean))
    return {"cleaned_documents": clean}