"""
study_mode_pipeline.py  —  Study Mode Pipeline

Study Mode is designed for learning from ingested documents.  Unlike Chat
(answer fast) or Deep Research (answer thoroughly), Study Mode structures
the retrieved knowledge into active learning artifacts:

    flashcards, quiz questions, concept maps, and a learning path.

Retrieval strategy
------------------
* Step 1 : Hybrid retrieval (Dense ALL-dims + BM25 + RRF) — same as deep
           research but with a wider top_k to maximise coverage.
* Step 2 : Contextual compression — trim irrelevant sentences per chunk.
* Step 3 : Cross-encoder rerank + score threshold — surface the most
           pedagogically relevant chunks.
* Step 4 : Graph augmentation (StudyModeRetriever) — add concept-relationship
           hops from the KnowledgeGraph for a richer learning path.
* Step 5 : Flashcard generation — LLM generates Q&A pairs from top chunks.
* Step 6 : Quiz generation — LLM generates MCQ questions from top chunks.
* Step 7 : Summary generation — concise study summary for the retrieved context.

All generation steps are optional; pass generate_flashcards=False etc. to skip.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Flashcard / Quiz helpers ───────────────────────────────────────────────────

_FLASHCARD_PROMPT = """\
You are a study assistant.  From the CONTEXT below, generate {n} concise flashcards.
Each flashcard must have:
  FRONT: a short question or term
  BACK:  a clear, accurate answer (1-3 sentences)

Output ONLY the flashcards in this exact format, one per block:
FRONT: <question>
BACK: <answer>

CONTEXT:
{context}
"""

_QUIZ_PROMPT = """\
You are a study assistant.  From the CONTEXT below, generate {n} multiple-choice
questions to test understanding.  Each question must have 4 options (A-D) with
exactly one correct answer.

Output ONLY the questions in this exact format:
Q: <question text>
A) <option>
B) <option>
C) <option>
D) <option>
ANSWER: <correct letter>

CONTEXT:
{context}
"""

_SUMMARY_PROMPT = """\
You are a study assistant.  Summarise the CONTEXT below into a clear, structured
study note (bullet points preferred).  Focus on key concepts, definitions, and
relationships.  Keep it under 300 words.

CONTEXT:
{context}

STUDY SUMMARY:
"""


def _parse_flashcards(raw: str) -> List[Dict[str, str]]:
    """Parse LLM flashcard output into [{front, back}, ...] list."""
    cards = []
    front = back = None
    for line in raw.strip().splitlines():
        line = line.strip()
        if line.startswith("FRONT:"):
            front = line[len("FRONT:"):].strip()
        elif line.startswith("BACK:"):
            back = line[len("BACK:"):].strip()
            if front and back:
                cards.append({"front": front, "back": back})
                front = back = None
    return cards


def _parse_quiz(raw: str) -> List[Dict[str, Any]]:
    """Parse LLM quiz output into [{question, options, answer}, ...] list."""
    questions = []
    current: Dict[str, Any] = {}
    for line in raw.strip().splitlines():
        line = line.strip()
        if line.startswith("Q:"):
            if current.get("question"):
                questions.append(current)
            current = {"question": line[2:].strip(), "options": {}}
        elif line[:2] in ("A)", "B)", "C)", "D)"):
            current.setdefault("options", {})[line[0]] = line[3:].strip()
        elif line.startswith("ANSWER:"):
            current["answer"] = line[len("ANSWER:"):].strip()
    if current.get("question"):
        questions.append(current)
    return questions


# ── StudyModePipeline ─────────────────────────────────────────────────────────

class StudyModePipeline:
    """
    Entry point for Study Mode.

    Parameters
    ----------
    hybrid_retriever     : HybridRetriever
    contextual_compressor: ContextualCompressor
    reranker             : Reranker  (BAAI/bge-reranker-base, lazy-loaded)
    llm                  : callable(prompt: str) -> str
    study_mode_retriever : StudyModeRetriever (graph augmentation, optional)
    top_k                : base retrieval count before compression/rerank (default 12)
    score_threshold      : drop reranked chunks below this score (default 0.0)
    n_flashcards         : flashcards to generate per run (default 5)
    n_quiz_questions     : MCQ questions to generate per run (default 5)
    generate_flashcards  : bool (default True)
    generate_quiz        : bool (default True)
    generate_summary     : bool (default True)
    """

    def __init__(
        self,
        hybrid_retriever,
        contextual_compressor,
        reranker,
        llm: Callable[[str], str],
        study_mode_retriever=None,
        top_k: int = 12,
        score_threshold: float = 0.0,
        n_flashcards: int = 5,
        n_quiz_questions: int = 5,
        generate_flashcards: bool = True,
        generate_quiz: bool = True,
        generate_summary: bool = True,
    ):
        self.retriever = hybrid_retriever
        self.compressor = contextual_compressor
        self.reranker = reranker
        self.llm = llm
        self.study_retriever = study_mode_retriever
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.n_flashcards = n_flashcards
        self.n_quiz_questions = n_quiz_questions
        self.do_flashcards = generate_flashcards
        self.do_quiz = generate_quiz
        self.do_summary = generate_summary

    def run(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Full study mode turn.

        Args:
            query          : topic / concept the user wants to study
            top_k          : override instance top_k for this turn
            score_threshold: override instance score_threshold for this turn

        Returns
        -------
        {
          "chunks"       : List[Dict]  — final chunks passed to generation
          "learning_path": List[Dict]  — concept hops from graph (if available)
          "flashcards"   : List[Dict]  — [{front, back}, ...]
          "quiz"         : List[Dict]  — [{question, options, answer}, ...]
          "summary"      : str         — concise study note
          "sources_used" : List[str]   — chunk ids used
        }
        """
        effective_top_k = top_k if top_k is not None else self.top_k
        effective_threshold = score_threshold if score_threshold is not None else self.score_threshold

        # ── Step 1: Hybrid retrieval ──────────────────────────────────
        # Wide initial fetch: top_k * 3 candidates for compression/rerank headroom
        candidate_k = effective_top_k * 3
        raw_chunks: List[Dict] = self.retriever.retrieve(query, top_k=candidate_k)

        # ── Step 2: Contextual compression ───────────────────────────
        # Strip each chunk to only the sentences relevant to the study topic.
        try:
            compressed = self.compressor.compress(raw_chunks, query)
        except Exception as exc:
            logger.warning("StudyMode[compress] failed, using raw chunks: %s", exc)
            compressed = raw_chunks

        # ── Step 3: Cross-encoder rerank + score threshold ────────────
        try:
            reranked = self.reranker.rerank(query, compressed, top_k=len(compressed))
            reranked = [
                c for c in reranked
                if c.get("rerank_score", 1.0) >= effective_threshold
            ]
            reranked = reranked[:effective_top_k]
        except Exception as exc:
            logger.warning("StudyMode[rerank] failed, using compressed chunks: %s", exc)
            reranked = compressed[:effective_top_k]

        logger.debug(
            "StudyMode: %d raw → %d compressed → %d reranked",
            len(raw_chunks), len(compressed), len(reranked),
        )

        # ── Step 4: Graph augmentation (optional) ─────────────────────
        # StudyModeRetriever adds concept-relationship hops from the
        # KnowledgeGraph — unique to study mode, gives a richer learning path.
        learning_path: List[Dict] = []
        if self.study_retriever is not None:
            try:
                graph_result = self.study_retriever.retrieve(query)
                # Merge unique graph chunks
                seen_ids = {c["id"] for c in reranked if "id" in c}
                for gc in graph_result.get("chunks", []):
                    if gc.get("id") not in seen_ids:
                        reranked.append(gc)
                        seen_ids.add(gc["id"])
                learning_path = graph_result.get("learning_path", [])
            except Exception as exc:
                logger.warning("StudyMode[graph_augment] failed: %s", exc)

        # Build context string for generation steps
        context = "\n\n---\n\n".join(c["content"] for c in reranked if c.get("content"))
        sources_used = [c["id"] for c in reranked if c.get("id")]

        # ── Step 5: Flashcard generation ──────────────────────────────
        flashcards: List[Dict[str, str]] = []
        if self.do_flashcards and context:
            try:
                raw = self.llm(
                    _FLASHCARD_PROMPT.format(n=self.n_flashcards, context=context)
                )
                flashcards = _parse_flashcards(raw)
            except Exception as exc:
                logger.warning("StudyMode[flashcards] LLM call failed: %s", exc)

        # ── Step 6: Quiz generation ────────────────────────────────────
        quiz: List[Dict[str, Any]] = []
        if self.do_quiz and context:
            try:
                raw = self.llm(
                    _QUIZ_PROMPT.format(n=self.n_quiz_questions, context=context)
                )
                quiz = _parse_quiz(raw)
            except Exception as exc:
                logger.warning("StudyMode[quiz] LLM call failed: %s", exc)

        # ── Step 7: Study summary ──────────────────────────────────────
        summary = ""
        if self.do_summary and context:
            try:
                summary = self.llm(_SUMMARY_PROMPT.format(context=context)).strip()
            except Exception as exc:
                logger.warning("StudyMode[summary] LLM call failed: %s", exc)

        return {
            "chunks":        reranked,
            "learning_path": learning_path,
            "flashcards":    flashcards,
            "quiz":          quiz,
            "summary":       summary,
            "sources_used":  sources_used,
        }
