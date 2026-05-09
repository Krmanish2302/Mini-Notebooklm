from typing import List, Dict, Any, Optional


class PromptBuilder:
    """Builds mode-specific prompts with citations and grounding.

    Method name contract (must match master_pipeline.py dispatch table):
        build_chat_prompt(query, documents, history="")     -> str
        build_study_prompt(query, documents, history="")    -> str
        build_research_prompt(query, documents, history="") -> str

    ``documents`` is a list of LangChain Document objects
    (each has .page_content and .metadata).  The context_window
    stored by ContextualEnricher is surfaced as
    metadata["context_window"] and is appended to the LLM context
    without being embedded.
    """

    GROUNDING_INSTRUCTION = (
        "CRITICAL INSTRUCTIONS:\n"
        "- Answer using ONLY the provided sources below.\n"
        "- Never make up information not in the sources.\n"
        "- If the answer is not in the sources, say \"I don't have enough information\".\n"
        "- Always cite sources using [SOURCE_X] format.\n"
        "- Be concise but complete.\n"
    )

    # ── Context formatting ────────────────────────────────────────────────────

    @staticmethod
    def format_context(documents: List[Any]) -> str:
        """Format LangChain Documents into a numbered SOURCE block.

        Includes context_window (surrounding sentences stored by
        ContextualEnricher) as a secondary paragraph so the LLM has
        richer surrounding context without it being embedded.
        """
        parts: List[str] = []
        for i, doc in enumerate(documents, 1):
            label = f"[SOURCE_{i}]"
            # Accept both LangChain Documents and plain dicts
            if hasattr(doc, "page_content"):
                content = doc.page_content
                meta = doc.metadata or {}
            else:
                content = doc.get("content", "")
                meta = {k: v for k, v in doc.items() if k != "content"}

            source_name = meta.get("source", meta.get("source_id", ""))
            header = f"{label} (from: {source_name})" if source_name else label

            block = f"{header}\n{content}"

            # Append context_window if present — shown to LLM, not embedded
            ctx_window = meta.get("context_window", "")
            if ctx_window:
                block += f"\n[SURROUNDING CONTEXT]\n{ctx_window}"

            parts.append(block)

        return "\n\n".join(parts)

    # ── Prompt builders (names must match master_pipeline dispatch table) ─────

    @staticmethod
    def build_chat_prompt(
        query: str,
        documents: List[Any],
        history: str = "",
    ) -> str:
        """Build prompt for Chat mode."""
        context = PromptBuilder.format_context(documents)
        history_block = f"CONVERSATION HISTORY:\n{history}\n\n" if history else ""
        return (
            f"{PromptBuilder.GROUNDING_INSTRUCTION}\n"
            f"{history_block}"
            f"RELEVANT SOURCES:\n{context}\n\n"
            f"USER QUESTION: {query}\n\n"
            f"ANSWER (with citations):"
        )

    @staticmethod
    def build_study_prompt(
        query: str,
        documents: List[Any],
        history: str = "",
        learning_path: Optional[List[Dict]] = None,
    ) -> str:
        """Build prompt for Study mode."""
        context = PromptBuilder.format_context(documents)
        history_block = f"CONVERSATION HISTORY:\n{history}\n\n" if history else ""

        path_block = ""
        if learning_path:
            steps = "\n".join(
                f"{i+1}. {s.get('from','')} → {s.get('to','')} ({s.get('relationship','related')})"
                for i, s in enumerate(learning_path[:5])
            )
            path_block = f"LEARNING PATH:\n{steps}\n\n"

        return (
            f"{PromptBuilder.GROUNDING_INSTRUCTION}\n"
            f"You are in Study Mode. Explain concepts clearly and show relationships.\n\n"
            f"{path_block}"
            f"{history_block}"
            f"RELEVANT SOURCES:\n{context}\n\n"
            f"USER QUESTION: {query}\n\n"
            f"EDUCATIONAL ANSWER (with citations and concept connections):"
        )

    @staticmethod
    def build_research_prompt(
        query: str,
        documents: List[Any],
        history: str = "",
    ) -> str:
        """Build prompt for Deep Research mode."""
        context = PromptBuilder.format_context(documents)
        history_block = f"CONVERSATION HISTORY:\n{history}\n\n" if history else ""
        return (
            f"{PromptBuilder.GROUNDING_INSTRUCTION}\n"
            f"You are in Deep Research mode. Provide a comprehensive, well-structured answer.\n\n"
            f"{history_block}"
            f"RELEVANT SOURCES (ranked by relevance):\n{context}\n\n"
            f"USER QUESTION: {query}\n\n"
            f"DETAILED ANSWER (with citations and analysis):"
        )

    # ── Backward-compat alias kept for any code still calling the old names ───

    @staticmethod
    def build_deep_research_prompt(
        query: str,
        documents: List[Any],
        history: str = "",
    ) -> str:
        """Alias → build_research_prompt."""
        return PromptBuilder.build_research_prompt(query, documents, history)

    @staticmethod
    def build_study_mode_prompt(
        query: str,
        documents: List[Any],
        learning_path: Optional[List[Dict]] = None,
        history: str = "",
    ) -> str:
        """Alias → build_study_prompt."""
        return PromptBuilder.build_study_prompt(query, documents, history, learning_path)
