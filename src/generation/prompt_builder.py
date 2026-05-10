"""
prompt_builder.py  —  Mode-specific prompt construction.

Persona (all modes)
-------------------
You're Carl Sagan if he were a chill classmate.
Grounded strictly in the retrieved sources ("our notebook").
Simple words, real-world analogies, poetic tone.
Short-to-medium answers.  If it’s not in the notes — say so.

Token philosophy
----------------
System prompts are kept intentionally short.  No redundant words.
Grounding is enforced via a single clean rule block, not a paragraph.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


# ── Shared persona + grounding block (reused across all modes) ────────────
_PERSONA = (
    "You're Carl Sagan if he were a chill classmate. "
    "Use simple words, real-world analogies, and a poetic touch. "
    "Answer only from the SOURCES below. "
    "If it’s not there, say: \"Not in my notes, bro.\""
)

# Grounding rules — minimal but airtight
_GROUND = (
    "Rules: "
    "(1) Use ONLY the sources. "
    "(2) Cite inline as [S1], [S2]… "
    "(3) Never invent facts."
)


class PromptBuilder:
    """
    Builds mode-specific prompts.

    Method name contract (must match master_pipeline.py dispatch table):
        build_chat_prompt(query, documents, history="")     -> str
        build_study_prompt(query, documents, history="")    -> str
        build_research_prompt(query, documents, history="") -> str
    """

    # ── Context formatting ────────────────────────────────────────────────────

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

    # ── Chat Mode ─────────────────────────────────────────────────────────

    @staticmethod
    def build_chat_prompt(
        query: str,
        documents: List[Any],
        history: str = "",
    ) -> str:
        """
        Chat mode — conversational, short-to-medium, strictly grounded.

        System: Carl Sagan persona + grounding rules.
        Keep answers concise — this is a quick back-and-forth, not a lecture.
        """
        ctx = PromptBuilder.format_context(documents)
        hist = f"HISTORY:\n{history}\n\n" if history else ""
        return (
            f"{_PERSONA}\n{_GROUND}\n\n"
            f"{hist}"
            f"SOURCES:\n{ctx}\n\n"
            f"Q: {query}\nA:"
        )

    # ── Study Mode ─────────────────────────────────────────────────────

    @staticmethod
    def build_study_prompt(
        query: str,
        documents: List[Any],
        history: str = "",
        learning_path: Optional[List[Dict]] = None,
    ) -> str:
        """
        Study mode — teach the concept clearly, show connections,
        use analogies.  Slightly longer than Chat but still tight.
        """
        ctx = PromptBuilder.format_context(documents)
        hist = f"HISTORY:\n{history}\n\n" if history else ""

        path_block = ""
        if learning_path:
            steps = " → ".join(
                f"{s.get('from', '')} ➔ {s.get('to', '')}"
                for s in learning_path[:4]
            )
            path_block = f"CONCEPT PATH: {steps}\n\n"

        return (
            f"{_PERSONA} You’re in teacher mode — build intuition, show how ideas connect.\n"
            f"{_GROUND}\n\n"
            f"{path_block}"
            f"{hist}"
            f"SOURCES:\n{ctx}\n\n"
            f"TOPIC: {query}\nEXPLAIN:"
        )

    # ── Deep Research Mode ─────────────────────────────────────────────

    @staticmethod
    def build_research_prompt(
        query: str,
        documents: List[Any],
        history: str = "",
    ) -> str:
        """
        Deep Research mode — thorough, structured, cited.
        Still Carl Sagan — wonder + precision, not dry academia.
        """
        ctx = PromptBuilder.format_context(documents)
        hist = f"HISTORY:\n{history}\n\n" if history else ""
        return (
            f"{_PERSONA} Go deep — structured, thorough, cite everything.\n"
            f"{_GROUND}\n\n"
            f"{hist}"
            f"SOURCES:\n{ctx}\n\n"
            f"RESEARCH Q: {query}\nDETAILED ANSWER:"
        )

    # ── Backward-compat aliases ───────────────────────────────────────

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
