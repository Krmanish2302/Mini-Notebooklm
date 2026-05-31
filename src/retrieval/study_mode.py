"""
study_mode.py

Study mode generates flashcards, quizzes, and summaries from retrieved docs
using configured open-source or custom LLM registry models.
"""
from __future__ import annotations
import logging
import os
import re
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)


def _get_llm():
    from src.generation.llm_registry import LLMRegistry
    return LLMRegistry.get(temperature=0.3)


def _docs_to_text(docs: List[Document], max_chars: int = 8000) -> str:
    return "\n\n---\n\n".join(d.page_content.strip() for d in docs)[:max_chars]


class StudyMode:
    """
    Generates study materials from retrieved documents.
    """

    # ── Flashcards ─────────────────────────────────────────────────────────

    _FLASHCARD_PROMPT = ChatPromptTemplate.from_messages([
        ("system",
         "Create 3 study flashcards from the context below. "
         "Format EXACTLY as:\nQ: <question>\nA: <answer>\n\n"
         "One flashcard per block, separated by blank lines."),
        ("human", "Context:\n{context}"),
    ])

    def flashcards(self, docs: List[Document]) -> List[Dict[str, str]]:
        try:
            chain = self._FLASHCARD_PROMPT | _get_llm() | StrOutputParser()
            raw   = chain.invoke({"context": _docs_to_text(docs)})
            cards = []
            for block in raw.strip().split("\n\n"):
                lines = [l.strip() for l in block.splitlines() if l.strip()]
                q = next((l.replace("Q:", "").strip() for l in lines if l.startswith("Q:")), "")
                a = next((l.replace("A:", "").strip() for l in lines if l.startswith("A:")), "")
                if q and a:
                    cards.append({"question": q, "answer": a, "difficulty": "Medium"})
            return cards
        except Exception as e:
            logger.warning("Failed to generate flashcards: %s", e)
            return []

    # ── Summary ─────────────────────────────────────────────────────────────

    _SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
        ("system",
         "Write a clear, concise summary of the following context. "
         "Preserve all key facts, figures, and concepts. "
         "Use bullet points for key takeaways at the end."),
        ("human", "Context:\n{context}"),
    ])

    def summary_bullets(self, docs: List[Document]) -> List[str]:
        try:
            chain = self._SUMMARY_PROMPT | _get_llm() | StrOutputParser()
            raw = chain.invoke({"context": _docs_to_text(docs)})
            bullets = []
            for line in raw.splitlines():
                line = line.strip()
                if line.startswith(("-", "•", "*")):
                    bullet_text = re.sub(r'^[-•*]\s*', '', line).strip()
                    if bullet_text:
                        bullets.append(bullet_text)
            if not bullets:
                # Fallback: split by sentences
                sentences = re.split(r'(?<=[.!?])\s+', raw.strip())
                bullets = [s.strip() for s in sentences if len(s.strip()) > 10][:4]
            return bullets
        except Exception as e:
            logger.warning("Failed to generate summary bullets: %s", e)
            return []