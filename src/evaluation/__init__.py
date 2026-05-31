"""
src/evaluation — RAGAS-style evaluation layer.

Usage:
    # Option A — direct call
    from src.evaluation import RAGASEvaluator, attach_ragas
    result = attach_ragas(pipeline_result, query="What is RAG?")

    # Option B — LangChain callback (auto-fires on every chain end)
    from src.evaluation import RAGASCallbackHandler
    handler = RAGASCallbackHandler()
    llm.invoke("...", config={"callbacks": [handler]})
    print(handler.last_result)

    # Option C — LCEL RunnableLambda
    from src.evaluation import make_eval_step
    chain = retrieval_chain | llm | make_eval_step(query="...")
"""
from .ragas_evaluator import RAGASEvaluator, RAGASResult   # noqa: F401
RagasEvaluator = RAGASEvaluator
from .ragas_bridge    import (                              # noqa: F401
    attach_ragas,
    get_default_evaluator,
    RAGASCallbackHandler,
    make_eval_step,
)

__all__ = [
    "RAGASEvaluator",
    "RagasEvaluator",
    "RAGASResult",
    "attach_ragas",
    "get_default_evaluator",
    "RAGASCallbackHandler",
    "make_eval_step",
]