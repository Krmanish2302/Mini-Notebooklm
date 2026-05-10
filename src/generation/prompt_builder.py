"""
prompt_builder.py  —  Mode-specific prompt construction.

Chat mode uses a PersonaConfig to build the system prompt.
Study + Research modes keep the Carl Sagan persona — they are
pipeline-internal and the user doesn't customise them.

Token philosophy
----------------
System prompts are kept intentionally short.  No redundant words.
Grounding is enforced via a single clean rule block.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.generation.persona_config import PersonaConfig


# ── Hardcoded personas for Study / Research (not user-facing) ─────────────────
_PERSONA_STUDY = (
    "You're Carl Sagan if he were a chill classmate in teacher mode. "
    "Build intuition, use analogies, show how ideas connect.  "
    "Use ONLY the sources. Cite as [S1], [S2]… "
    "If it's not there, say: 'Not in my notes, bro.'"
)

_PERSONA_RESEARCH = (
    "You're Carl Sagan if he were a chill classmate. "
    "Go deep — structured, thorough, cite everything as [S1], [S2]…  "
    "Use ONLY the sources. Never invent facts. "
    "If it's not there, say: 'Not in my notes, bro.'"
)

_GROUNDING = (
    "Rules: "
    "(1) Use ONLY the sources.  "
    "(2) Cite inline as [S1], [S2]…  "
    "(3) Never invent facts."
)


class PromptBuilder:
    """
    Builds mode-specific prompts.

    Method contract (must match master_pipeline.py dispatch table):
        build_chat_prompt(query, documents, history="", persona_config=None)  -> str
        build_study_prompt(query, documents, history="")                       -> str
        build_research_prompt(query, documents, history="")                    -> str
    """

    # ── Context formatting ─────────────────────────────────────────────────────

    @staticmethod
    def format_context(documents: List[Any]) -> str:
        """
        Format retrieved chunks into a compact numbered SOURCE block.
        Appends context_window (from ContextualEnricher) when present.
        """
        parts: List[str] = []
        for i, doc in enumerate(documents, 1):
            if hasattr(doc, "page_content"):
                content = doc.page_content
                meta = doc.metadata or {}
            else:
                content = doc.get("content", "")
                meta = {k: v for k, v in doc.items() if k != "content"}

            src = meta.get("source", meta.get("source_id", ""))
            header = f"[S{i}]{' — ' + src if src else ''}"
            block = f"{header}\n{content}"

            ctx = meta.get("context_window", "")
            if ctx:
                block += f"\n[context] {ctx}"

            parts.append(block)

        return "\n\n".join(parts)

    # ── Chat Mode ──────────────────────────────────────────────────────────────

    @staticmethod
    def build_chat_prompt(
        query: str,
        documents: List[Any],
        history: str = "",
        persona_config: Optional[PersonaConfig] = None,
    ) -> str:
        """
        Chat mode — persona is driven by PersonaConfig.
        Falls back to the default Carl Sagan preset when no config is given.
        Grounding is ALWAYS present regardless of persona choice.
        """
        cfg = persona_config or PersonaConfig()   # default: sagan / neutral / medium
        system = cfg.build_system_prompt()

        ctx = PromptBuilder.format_context(documents)
        hist = f"HISTORY:\n{history}\n\n" if history else ""
        return (
            f"{system}\n\n"
            f"{hist}"
            f"SOURCES:\n{ctx}\n\n"
            f"Q: {query}\nA:"
        )

    # ── Study Mode ─────────────────────────────────────────────────────────────

    @staticmethod
    def build_study_prompt(
        query: str,
        documents: List[Any],
        history: str = "",
        learning_path: Optional[List[Dict]] = None,
    ) -> str:
        """Study mode — fixed Sagan teacher persona, shows concept connections."""
        ctx = PromptBuilder.format_context(documents)
        hist = f"HISTORY:\n{history}\n\n" if history else ""

        path_block = ""
        if learning_path:
            steps = " → ".join(
                f"{s.get('from', '')} ➜ {s.get('to', '')}"
                for s in learning_path[:4]
            )
            path_block = f"CONCEPT PATH: {steps}\n\n"

        return (
            f"{_PERSONA_STUDY}\n\n"
            f"{path_block}"
            f"{hist}"
            f"SOURCES:\n{ctx}\n\n"
            f"TOPIC: {query}\nEXPLAIN:"
        )

    # ── Deep Research Mode ─────────────────────────────────────────────────────

    @staticmethod
    def build_research_prompt(
        query: str,
        documents: List[Any],
        history: str = "",
    ) -> str:
        """Deep Research — fixed Sagan research persona, thorough + cited."""
        ctx = PromptBuilder.format_context(documents)
        hist = f"HISTORY:\n{history}\n\n" if history else ""
        return (
            f"{_PERSONA_RESEARCH}\n\n"
            f"{hist}"
            f"SOURCES:\n{ctx}\n\n"
            f"RESEARCH Q: {query}\nDETAILED ANSWER:"
        )

    # ── Backward-compat aliases ────────────────────────────────────────────────

    @staticmethod
    def build_deep_research_prompt(
        query: str, documents: List[Any], history: str = ""
    ) -> str:
        return PromptBuilder.build_research_prompt(query, documents, history)

    @staticmethod
    def build_study_mode_prompt(
        query: str,
        documents: List[Any],
        learning_path: Optional[List[Dict]] = None,
        history: str = "",
    ) -> str:
        return PromptBuilder.build_study_prompt(query, documents, history, learning_path)
