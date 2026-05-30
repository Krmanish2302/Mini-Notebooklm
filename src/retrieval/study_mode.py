"""
study_mode.py

Study mode generates flashcards, quizzes, and summaries from retrieved docs
using LangChain LCEL chains.

Usage:
    from src.retrieval.study_mode import StudyMode
    from src.retrieval.advanced_retriever import AdvancedRetriever

    retriever = AdvancedRetriever("data/vectorstores/rep_001")
    result    = retriever.retrieve("machine learning basics")
    sm        = StudyMode()

    flashcards = sm.flashcards(result["documents"])
    quiz       = sm.quiz(result["documents"])
    summary    = sm.summary(result["documents"])
"""
from __future__ import annotations
import logging
import os
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)
STUDY_MODEL = os.getenv("STUDY_MODE_MODEL", "gpt-4o-mini")


def _get_llm():
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=STUDY_MODEL, temperature=0.3)


def _docs_to_text(docs: List[Document], max_chars: int = 8000) -> str:
    return "\n\n---\n\n".join(d.page_content.strip() for d in docs)[:max_chars]


class StudyMode:
    """
    Generates study materials from retrieved documents using LCEL chains.

    Methods:
        flashcards(docs)  → List[{"front": str, "back": str}]
        quiz(docs)        → List[{"question": str, "options": List[str], "answer": str}]
        summary(docs)     → str
    """

    # ── Flashcards ─────────────────────────────────────────────────────────

    _FLASHCARD_PROMPT = ChatPromptTemplate.from_messages([
        ("system",
         "Create 5 study flashcards from the context below. "
         "Format EXACTLY as:\nFRONT: <question>\nBACK: <answer>\n\n"
         "One flashcard per block, separated by blank lines."),
        ("human", "Context:\n{context}"),
    ])

    def flashcards(self, docs: List[Document]) -> List[Dict[str, str]]:
        chain = self._FLASHCARD_PROMPT | _get_llm() | StrOutputParser()
        raw   = chain.invoke({"context": _docs_to_text(docs)})
        cards = []
        for block in raw.strip().split("\n\n"):
            lines = [l.strip() for l in block.splitlines() if l.strip()]
            front = next((l.replace("FRONT:", "").strip() for l in lines if l.startswith("FRONT:")), "")
            back  = next((l.replace("BACK:",  "").strip() for l in lines if l.startswith("BACK:")),  "")
            if front and back:
                cards.append({"front": front, "back": back})
        return cards

    # ── Quiz ────────────────────────────────────────────────────────────────

    _QUIZ_PROMPT = ChatPromptTemplate.from_messages([
        ("system",
         "Create 5 multiple-choice quiz questions from the context below. "
         "Format EXACTLY as:\n"
         "Q: <question>\nA) <option1>\nB) <option2>\nC) <option3>\nD) <option4>\nANSWER: <A|B|C|D>\n\n"
         "One question per block, separated by blank lines."),
        ("human", "Context:\n{context}"),
    ])

    def quiz(self, docs: List[Document]) -> List[Dict[str, Any]]:
        chain     = self._QUIZ_PROMPT | _get_llm() | StrOutputParser()
        raw       = chain.invoke({"context": _docs_to_text(docs)})
        questions = []
        for block in raw.strip().split("\n\n"):
            lines   = [l.strip() for l in block.splitlines() if l.strip()]
            q       = next((l.replace("Q:", "").strip() for l in lines if l.startswith("Q:")), "")
            options = [l for l in lines if l.startswith(("A)", "B)", "C)", "D)"))]
            answer  = next((l.replace("ANSWER:", "").strip() for l in lines if l.startswith("ANSWER:")), "")
            if q and options:
                questions.append({"question": q, "options": options, "answer": answer})
        return questions

    # ── Summary ─────────────────────────────────────────────────────────────

    _SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
        ("system",
         "Write a clear, concise summary of the following context. "
         "Preserve all key facts, figures, and concepts. "
         "Use bullet points for key takeaways at the end."),
        ("human", "Context:\n{context}"),
    ])

    def summary(self, docs: List[Document]) -> str:
        chain = self._SUMMARY_PROMPT | _get_llm() | StrOutputParser()
        return chain.invoke({"context": _docs_to_text(docs)})