"""
ragas_evaluator.py — RAGAS-style evaluation for Mini NotebookLM.

Computes 5 core RAGAS metrics LOCALLY (no external RAGAS library required).
All metrics return a float in [0.0, 1.0].

Metrics
-------
1. Faithfulness (Grounding Score)
   Sentence-level token-overlap NLI.
   Score = supported_sentences / total_answer_sentences

2. Answer Relevance
   Cosine similarity between question and answer embeddings.

3. Context Precision
   Fraction of retrieved chunks that contributed to the answer.

4. Context Recall  [ground_truth required]
   Fraction of ground-truth sentences covered by the retrieved context.

5. Answer Similarity  [ground_truth required]
   Cosine similarity between generated answer and ground-truth embeddings.

LangChain integration
---------------------
- context_chunks accept List[Document] OR List[dict] — both work.
- evaluate_batch() returns a Pandas DataFrame when pandas is installed.
- async_evaluate() is non-blocking via asyncio.to_thread.

Usage
-----
    evaluator = RAGASEvaluator()
    result = evaluator.evaluate(
        question="What is RAG?",
        answer="RAG stands for Retrieval-Augmented Generation...",
        context_chunks=[Document(page_content="...")],
        ground_truth=None,
    )
    print(result.faithfulness)       # 0.87
    print(result.overall_score)      # 0.81
    print(result.grade)              # "Good"
    print(result.to_dict())          # JSON-serialisable dict
"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# ── Chunk normalisation ────────────────────────────────────────────────────────

def _as_dict(chunk: Union[Document, Dict[str, Any]]) -> Dict[str, Any]:
    """Accept both LangChain Documents and raw dicts transparently."""
    if isinstance(chunk, Document):
        d = dict(chunk.metadata)
        d["content"] = chunk.page_content
        return d
    return chunk


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class RAGASResult:
    faithfulness:        float
    answer_relevance:    float
    context_precision:   float
    context_recall:      Optional[float]   # None when no ground_truth
    answer_similarity:   Optional[float]   # None when no ground_truth
    # meta
    question:            str
    answer_sentences:    int
    supported_sentences: int
    chunks_evaluated:    int
    chunks_contributed:  int
    has_ground_truth:    bool
    chunk_details:       List[Dict[str, Any]]

    # ── Composite score ───────────────────────────────────────────────────────

    @property
    def overall_score(self) -> float:
        """
        Weighted composite.
        With ground truth:    faith 0.35 + relevance 0.30 + precision 0.20 + recall 0.15
        Without ground truth: faith 0.40 + relevance 0.35 + precision 0.25
        """
        if self.has_ground_truth and self.context_recall is not None:
            return (
                0.35 * self.faithfulness
                + 0.30 * self.answer_relevance
                + 0.20 * self.context_precision
                + 0.15 * self.context_recall
            )
        return (
            0.40 * self.faithfulness
            + 0.35 * self.answer_relevance
            + 0.25 * self.context_precision
        )

    @property
    def grade(self) -> str:
        s = self.overall_score
        if s >= 0.85: return "Excellent"
        if s >= 0.70: return "Good"
        if s >= 0.55: return "Fair"
        return "Poor"

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["overall_score"]       = round(self.overall_score, 4)
        d["grade"]               = self.grade
        d["faithfulness"]        = round(self.faithfulness, 4)
        d["answer_relevance"]    = round(self.answer_relevance, 4)
        d["context_precision"]   = round(self.context_precision, 4)
        if self.context_recall is not None:
            d["context_recall"]  = round(self.context_recall, 4)
        if self.answer_similarity is not None:
            d["answer_similarity"] = round(self.answer_similarity, 4)
        return d

    def to_dataframe(self):
        """
        Returns a single-row Pandas DataFrame.
        Useful for batch evaluation aggregation.
        Requires pandas.
        """
        import pandas as pd  # type: ignore
        row = {k: v for k, v in self.to_dict().items() if k != "chunk_details"}
        return pd.DataFrame([row])

    def __repr__(self) -> str:
        return (
            f"RAGASResult(grade={self.grade!r}, overall={self.overall_score:.3f}, "
            f"faithfulness={self.faithfulness:.3f}, relevance={self.answer_relevance:.3f})"
        )


# ── Evaluator ─────────────────────────────────────────────────────────────────

class RAGASEvaluator:
    """
    Local RAGAS-style evaluator.

    Parameters
    ----------
    embedding_model   : SentenceTransformers model name
    overlap_threshold : min token-overlap ratio for a sentence to be "supported"
    """

    def __init__(
        self,
        embedding_model:   str   = "all-MiniLM-L6-v2",
        overlap_threshold: float = 0.25,
    ):
        self._emb_model_name   = embedding_model
        self._embedder         = None        # lazy-loaded on first use
        self.overlap_threshold = overlap_threshold

    # ── Lazy embedder ─────────────────────────────────────────────────────────

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self._emb_model_name)
        return self._embedder

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        question:       str,
        answer:         str,
        context_chunks: List[Union[Document, Dict[str, Any]]],
        ground_truth:   Optional[str] = None,
    ) -> RAGASResult:
        """
        Run all applicable metrics. Accepts Documents or dicts as chunks.
        Fast enough to run synchronously after every LLM response.
        """
        if not answer or not answer.strip():
            return self._empty_result(question, context_chunks)

        chunks         = [_as_dict(c) for c in context_chunks]
        answer_sents   = self._split_sentences(answer)
        context_text   = self._chunks_to_text(chunks)

        # 1. Faithfulness
        supported, chunk_details = self._faithfulness(
            answer_sents, context_text, chunks
        )

        # 2. Answer Relevance
        relevance = self._answer_relevance(question, answer)

        # 3. Context Precision
        precision, contributed = self._context_precision(answer_sents, chunks)

        # 4 & 5. Ground-truth metrics (optional)
        recall     = None
        similarity = None
        if ground_truth and ground_truth.strip():
            recall     = self._context_recall(ground_truth, context_text)
            similarity = self._answer_similarity(answer, ground_truth)

        return RAGASResult(
            faithfulness        = len(supported) / len(answer_sents) if answer_sents else 0.0,
            answer_relevance    = relevance,
            context_precision   = precision,
            context_recall      = recall,
            answer_similarity   = similarity,
            question            = question,
            answer_sentences    = len(answer_sents),
            supported_sentences = len(supported),
            chunks_evaluated    = len(chunks),
            chunks_contributed  = contributed,
            has_ground_truth    = bool(ground_truth and ground_truth.strip()),
            chunk_details       = chunk_details,
        )

    async def async_evaluate(
        self,
        question:       str,
        answer:         str,
        context_chunks: List[Union[Document, Dict[str, Any]]],
        ground_truth:   Optional[str] = None,
    ) -> RAGASResult:
        """
        Non-blocking wrapper — runs evaluate() in a thread pool.
        Use in async FastAPI handlers to avoid blocking the event loop.
        """
        return await asyncio.to_thread(
            self.evaluate, question, answer, context_chunks, ground_truth
        )

    def evaluate_batch(
        self,
        samples: List[Dict[str, Any]],
    ) -> List[RAGASResult]:
        """
        Evaluate a list of samples.
        Each sample dict: {question, answer, context_chunks, ground_truth?}
        Returns List[RAGASResult].

        To get a summary DataFrame:
            import pandas as pd
            results = evaluator.evaluate_batch(samples)
            df = pd.concat([r.to_dataframe() for r in results], ignore_index=True)
            print(df.describe())
        """
        results = []
        for i, s in enumerate(samples):
            try:
                r = self.evaluate(
                    question=s["question"],
                    answer=s["answer"],
                    context_chunks=s.get("context_chunks", []),
                    ground_truth=s.get("ground_truth"),
                )
                results.append(r)
            except Exception as exc:
                logger.warning("[evaluate_batch] Sample %d failed: %s", i, exc)
                results.append(self._empty_result(s.get("question", ""), []))
        return results

    # ── Metric 1: Faithfulness ────────────────────────────────────────────────

    def _faithfulness(
        self,
        answer_sents:   List[str],
        context_text:   str,
        context_chunks: List[Dict],
    ):
        ctx_tokens = self._tokenise(context_text)
        supported  = []
        sent_details: List[Dict] = []

        for i, sent in enumerate(answer_sents):
            sent_tokens = self._tokenise(sent)
            if not sent_tokens:
                continue
            overlap      = len(sent_tokens & ctx_tokens) / len(sent_tokens)
            is_supported = overlap >= self.overlap_threshold
            if is_supported:
                supported.append(i)
            sent_details.append({
                "sentence":  sent,
                "overlap":   round(overlap, 3),
                "supported": is_supported,
            })

        # Per-chunk contribution
        chunk_details = []
        for c in context_chunks:
            c_tok   = self._tokenise(c.get("content", ""))
            contrib = sum(
                1 for s in answer_sents
                if c_tok and
                   len(self._tokenise(s) & c_tok) / max(len(self._tokenise(s)), 1)
                   >= self.overlap_threshold
            )
            cid = c.get("id", c.get("chunk_id", c.get("source_id", "?")))
            cid_str = str(cid) if cid is not None else "?"
            chunk_details.append({
                "chunk_id":           cid_str[:16],
                "source":             c.get("source_id", "unknown"),
                "citation":           c.get("citation_label", ""),
                "contributed":        contrib > 0,
                "sentences_supported": contrib,
                "score":              round(contrib / max(len(answer_sents), 1), 3),
            })

        return supported, chunk_details

    # ── Metric 2: Answer Relevance ────────────────────────────────────────────

    def _answer_relevance(self, question: str, answer: str) -> float:
        try:
            vecs = self._get_embedder().encode(
                [question, answer], normalize_embeddings=True
            )
            return max(0.0, float(np.dot(vecs[0], vecs[1])))
        except Exception as exc:
            logger.warning("[answer_relevance] Embedding failed, using overlap: %s", exc)
            q_tok = self._tokenise(question)
            a_tok = self._tokenise(answer)
            return len(q_tok & a_tok) / len(q_tok) if q_tok else 0.0

    # ── Metric 3: Context Precision ───────────────────────────────────────────

    def _context_precision(
        self,
        answer_sents:   List[str],
        context_chunks: List[Dict],
    ):
        if not context_chunks:
            return 0.0, 0
        answer_tokens: set = set()
        for s in answer_sents:
            answer_tokens |= self._tokenise(s)

        contributed = 0
        for c in context_chunks:
            c_tok = self._tokenise(c.get("content", ""))
            if c_tok and len(c_tok & answer_tokens) / len(c_tok) >= self.overlap_threshold:
                contributed += 1

        return contributed / len(context_chunks), contributed

    # ── Metric 4: Context Recall ──────────────────────────────────────────────

    def _context_recall(self, ground_truth: str, context_text: str) -> float:
        gt_sents   = self._split_sentences(ground_truth)
        ctx_tokens = self._tokenise(context_text)
        if not gt_sents:
            return 0.0
        covered = sum(
            1 for s in gt_sents
            if len(self._tokenise(s) & ctx_tokens) / max(len(self._tokenise(s)), 1)
               >= self.overlap_threshold
        )
        return covered / len(gt_sents)

    # ── Metric 5: Answer Similarity ───────────────────────────────────────────

    def _answer_similarity(self, answer: str, ground_truth: str) -> float:
        try:
            vecs = self._get_embedder().encode(
                [answer, ground_truth], normalize_embeddings=True
            )
            return max(0.0, float(np.dot(vecs[0], vecs[1])))
        except Exception as exc:
            logger.warning("[answer_similarity] Embedding failed, using overlap: %s", exc)
            a_tok  = self._tokenise(answer)
            gt_tok = self._tokenise(ground_truth)
            return len(a_tok & gt_tok) / len(gt_tok) if gt_tok else 0.0

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        if not text:
            return []
        raw = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in raw if len(s.strip()) > 8]

    @staticmethod
    def _tokenise(text: str) -> set:
        _STOP = {
            "a","an","the","is","it","in","on","at","to","of","for",
            "and","or","but","not","with","this","that","are","was",
            "be","by","from","as","has","have","its","their","they",
            "we","you","i","he","she","do","did","can","will","about",
        }
        return {
            t for t in re.findall(r"[a-z]+", text.lower())
            if t not in _STOP and len(t) > 2
        }

    @staticmethod
    def _chunks_to_text(chunks: List[Dict]) -> str:
        return " ".join(c.get("content", "") for c in chunks)

    def _empty_result(
        self,
        question:       str,
        context_chunks: List,
    ) -> RAGASResult:
        return RAGASResult(
            faithfulness=0.0, answer_relevance=0.0,
            context_precision=0.0, context_recall=None, answer_similarity=None,
            question=question, answer_sentences=0, supported_sentences=0,
            chunks_evaluated=len(context_chunks), chunks_contributed=0,
            has_ground_truth=False, chunk_details=[],
        )