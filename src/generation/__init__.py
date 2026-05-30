"""
src/generation/__init__.py

Public API for the generation package.

Usage:
    from src.generation import generate

    result = generate(
        query="What are the main conclusions?",
        documents=docs,          # List[Document] from retrieval
        mode="chat",             # "chat" | "study" | "research"
    )
    print(result["answer"])
    print(result["citations"])
"""
from .generation_graph import generation_app       # noqa: F401
from .state            import GenerationState      # noqa: F401
from .persona_config   import PersonaConfig        # noqa: F401
from .llm_registry     import LLMRegistry          # noqa: F401


def generate(
    query:     str,
    documents: list,
    mode:      str  = "chat",
    history:   str  = "",
    persona:   "PersonaConfig | None" = None,
    stream:    bool = False,
) -> dict:
    """
    One-line entry point.

    Returns final GenerationState dict:
        answer, citations, follow_ups, sources_used, chunks_used, tokens_estimate
    """
    return generation_app.invoke(GenerationState(
        query=query,
        documents=documents,
        mode=mode,
        history=history,
        persona=persona or PersonaConfig(),
        stream=stream,
    ))


__all__ = ["generate", "generation_app", "GenerationState", "PersonaConfig", "LLMRegistry"]