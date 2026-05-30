"""
ragas_bridge.py — LangChain-native bridge between pipelines and RAGASEvaluator.

Three integration patterns:

  Pattern A — Manual (original, preserved):
      result = attach_ragas(pipeline_result, query="What is RAG?")

  Pattern B — LangChain BaseCallbackHandler (auto-fires on chain end):
      handler = RAGASCallbackHandler(query="What is RAG?", context_chunks=docs)
      llm.invoke("...", config={"callbacks": [handler]})
      print(handler.last_result)    # RAGASResult

  Pattern C — LCEL RunnableLambda (inline in chains):
      from src.evaluation import make_eval_step
      chain = retrieval_chain | llm | make_eval_step(query="What is RAG?")
      result = chain.invoke({"question": "What is RAG?"})
      # result["ragas"] is now populated

Keys read from result dict (Pattern A):
    answer, context_chunks | chunks_used | retrieved_chunks

Keys written to result dict (all patterns):
    ragas : dict  — RAGASResult.to_dict() output
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from src.evaluation.ragas_evaluator import RAGASEvaluator, RAGASResult

logger = logging.getLogger(__name__)

# ── Singleton evaluator ────────────────────────────────────────────────────────

_DEFAULT_EVALUATOR: Optional[RAGASEvaluator] = None


def get_default_evaluator() -> RAGASEvaluator:
    global _DEFAULT_EVALUATOR
    if _DEFAULT_EVALUATOR is None:
        _DEFAULT_EVALUATOR = RAGASEvaluator()
    return _DEFAULT_EVALUATOR


# ── Pattern A — Manual attach ─────────────────────────────────────────────────

def attach_ragas(
    result:       Dict[str, Any],
    query:        str,
    evaluator:    Optional[RAGASEvaluator] = None,
    ground_truth: Optional[str]            = None,
) -> Dict[str, Any]:
    """
    Evaluate the pipeline result dict and attach a 'ragas' key in-place.

    Parameters
    ----------
    result       : pipeline result dict (mutated in-place and returned)
    query        : original user question
    evaluator    : RAGASEvaluator instance (uses singleton if None)
    ground_truth : optional reference answer

    Returns
    -------
    Same result dict with 'ragas' key added (dict or None on failure).
    """
    if evaluator is None:
        try:
            evaluator = get_default_evaluator()
        except Exception as exc:
            logger.warning("[attach_ragas] Could not load evaluator: %s", exc)
            result["ragas"] = None
            return result

    answer: str = result.get("answer", "")
    if not answer or not answer.strip():
        result["ragas"] = None
        return result

    # Resolve context chunks — prefer deduplicated/labelled list
    ctx_chunks: List[Any] = (
        result.get("context_chunks")
        or result.get("chunks_used")
        or result.get("retrieved_chunks", [])
    )[:20]   # cap at 20 to keep evaluation fast

    if not ctx_chunks:
        logger.debug("[attach_ragas] No context chunks — skipping evaluation")
        result["ragas"] = None
        return result

    try:
        ragas_result: RAGASResult = evaluator.evaluate(
            question=query,
            answer=answer,
            context_chunks=ctx_chunks,
            ground_truth=ground_truth,
        )
        result["ragas"] = ragas_result.to_dict()
        logger.debug(
            "[attach_ragas] faithfulness=%.3f relevance=%.3f overall=%.3f grade=%s",
            ragas_result.faithfulness,
            ragas_result.answer_relevance,
            ragas_result.overall_score,
            ragas_result.grade,
        )
    except Exception as exc:
        logger.warning("[attach_ragas] Evaluation failed: %s", exc)
        result["ragas"] = None

    return result


# ── Pattern B — LangChain BaseCallbackHandler ──────────────────────────────────

class RAGASCallbackHandler(BaseCallbackHandler):
    """
    LangChain callback that auto-evaluates after every chain end.

    Attach to any LLM / chain / agent run:
        handler = RAGASCallbackHandler(
            query="What is photosynthesis?",
            context_chunks=retrieved_docs,
        )
        chain.invoke(input, config={"callbacks": [handler]})
        result = handler.last_result
        print(result.grade)

    The handler fires on on_chain_end and stores the last RAGASResult
    in self.last_result and self.history.
    """

    def __init__(
        self,
        query:          str,
        context_chunks: Optional[List[Union[Document, Dict]]] = None,
        ground_truth:   Optional[str]                         = None,
        evaluator:      Optional[RAGASEvaluator]              = None,
    ):
        super().__init__()
        self.query          = query
        self.context_chunks = context_chunks or []
        self.ground_truth   = ground_truth
        self.evaluator      = evaluator or get_default_evaluator()
        self.last_result:   Optional[RAGASResult] = None
        self.history:       List[RAGASResult]     = []

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called automatically when any LangChain chain finishes."""
        # Extract answer from output — handles str, dict, AIMessage
        answer = ""
        if isinstance(outputs, str):
            answer = outputs
        elif isinstance(outputs, dict):
            answer = (
                outputs.get("answer")
                or outputs.get("text")
                or outputs.get("output")
                or outputs.get("content", "")
            )
        elif hasattr(outputs, "content"):
            answer = outputs.content

        if not answer or not isinstance(answer, str):
            return

        try:
            result = self.evaluator.evaluate(
                question=self.query,
                answer=answer,
                context_chunks=self.context_chunks,
                ground_truth=self.ground_truth,
            )
            self.last_result = result
            self.history.append(result)
            logger.debug(
                "[RAGASCallbackHandler] grade=%s overall=%.3f",
                result.grade, result.overall_score,
            )
        except Exception as exc:
            logger.warning("[RAGASCallbackHandler] Evaluation failed: %s", exc)

    def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        """Also fires on raw LLM responses (not wrapped in a chain)."""
        try:
            answer = response.generations[0][0].text
        except Exception:
            return
        if answer:
            self.on_chain_end({"answer": answer}, run_id=run_id)

    def summary(self) -> Dict[str, Any]:
        """Aggregate stats over all evaluations in this session."""
        if not self.history:
            return {"count": 0}
        import statistics
        scores = [r.overall_score for r in self.history]
        return {
            "count":              len(self.history),
            "avg_overall":        round(statistics.mean(scores), 4),
            "avg_faithfulness":   round(statistics.mean(r.faithfulness for r in self.history), 4),
            "avg_relevance":      round(statistics.mean(r.answer_relevance for r in self.history), 4),
            "grade_distribution": {
                g: sum(1 for r in self.history if r.grade == g)
                for g in ("Excellent", "Good", "Fair", "Poor")
            },
        }


# ── Pattern C — LCEL RunnableLambda ───────────────────────────────────────────

def make_eval_step(
    query:          str,
    context_chunks: Optional[List[Union[Document, Dict]]] = None,
    ground_truth:   Optional[str]                         = None,
    evaluator:      Optional[RAGASEvaluator]              = None,
    key:            str                                   = "ragas",
) -> RunnableLambda:
    """
    Returns a RunnableLambda that evaluates the answer and injects
    the RAGAS result into the pipeline dict under *key*.

    Usage in an LCEL chain:
        chain = (
            retrieval_step
            | generation_step
            | make_eval_step(query=question, context_chunks=docs)
        )
        output = chain.invoke({"question": question})
        print(output["ragas"]["grade"])
    """
    _eval = evaluator or get_default_evaluator()
    _ctx  = context_chunks or []

    def _run(result: Any) -> Any:
        # result may be dict, str, or AIMessage
        if isinstance(result, str):
            result = {"answer": result}
        elif hasattr(result, "content"):
            result = {"answer": result.content}
        result = dict(result)   # shallow copy

        answer = result.get("answer", "")
        ctx    = _ctx or result.get("context_chunks") or result.get("chunks_used", [])

        if answer and ctx:
            try:
                r = _eval.evaluate(
                    question=query,
                    answer=answer,
                    context_chunks=ctx[:20],
                    ground_truth=ground_truth,
                )
                result[key] = r.to_dict()
            except Exception as exc:
                logger.warning("[make_eval_step] Evaluation failed: %s", exc)
                result[key] = None
        else:
            result[key] = None

        return result

    return RunnableLambda(_run)