"""
persona_config.py  —  Persona + response-style configuration for Chat mode.

Design decision
---------------
We use PRESETS (curated, tested personas) as the primary UX —
not a free-text input field — because:

  1. Token efficiency  — presets are <15 words each; a free textarea lets
     users write 200-word prompts that balloon cost with no quality gain.
  2. Grounding safety  — every preset includes the strict source-only rule
     baked in.  A free field could accidentally override grounding.
  3. Better UX         — dropdown + tone slider is faster than writing a
     custom system prompt from scratch.
  4. Power-user escape hatch  — `custom_persona` field allows full override
     when the user knows what they are doing (advanced toggle in UI).

Preset taxonomy
---------------
Persona  — *who* the assistant sounds like (character / role)
Tone     — *how* it sounds (energy level, formality)
Length   — *how much* it writes

All three combine into a single system prompt string at build time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


# ── Preset catalogues ──────────────────────────────────────────────────────────

PERSONA_PRESETS: Dict[str, str] = {
    # key: short system-prompt fragment injected as the "who" line
    "sagan":       "You're Carl Sagan if he were a chill classmate — poetic, curious, full of wonder.",
    "professor":   "You're a clear-headed university professor — precise, structured, no fluff.",
    "eli5":        "You're explaining to a curious 12-year-old — simple words, fun analogies.",
    "analyst":     "You're a sharp research analyst — data-driven, concise, bullet-friendly.",
    "socratic":    "You're a Socratic tutor — answer questions with guiding questions, build insight step by step.",
    "journalist":  "You're an investigative journalist — lead with the key finding, then unpack.",
    "custom":      "{custom}",   # placeholder — replaced at build time with user text
}

TONE_PRESETS: Dict[str, str] = {
    "casual":      "Keep it conversational and friendly.",
    "neutral":     "",   # default — no tone modifier added
    "formal":      "Use formal, professional language.",
    "enthusiastic":"Be energetic and upbeat — make ideas feel exciting.",
    "stoic":       "Be calm, measured, and to the point.",
}

LENGTH_PRESETS: Dict[str, str] = {
    "short":   "Keep answers to 2-3 sentences max.",
    "medium":  "",   # default — no length modifier (natural length)
    "long":    "Give thorough, detailed answers with examples.",
    "bullets": "Always respond in bullet points.",
}

# Grounding rule — NEVER removed, regardless of persona override
_GROUNDING = (
    "Use ONLY the sources below. "
    "Cite as [S1], [S2]… "
    "If it's not there, say: 'Not in my notes, bro.'"
)


# ── Config dataclass ───────────────────────────────────────────────────────────

@dataclass
class PersonaConfig:
    """
    Holds the current persona + style settings for Chat mode.

    Attributes
    ----------
    persona       : key from PERSONA_PRESETS (default: "sagan")
    tone          : key from TONE_PRESETS    (default: "neutral")
    length        : key from LENGTH_PRESETS  (default: "medium")
    custom_persona: raw string — used when persona=="custom" (power-user override)
    """
    persona:        str = "sagan"
    tone:           str = "neutral"
    length:         str = "medium"
    custom_persona: Optional[str] = None

    # ── Validation ─────────────────────────────────────────────────────────────

    def __post_init__(self) -> None:
        if self.persona not in PERSONA_PRESETS:
            raise ValueError(
                f"Unknown persona '{self.persona}'. "
                f"Choose from: {list(PERSONA_PRESETS)}"
            )
        if self.tone not in TONE_PRESETS:
            raise ValueError(
                f"Unknown tone '{self.tone}'. "
                f"Choose from: {list(TONE_PRESETS)}"
            )
        if self.length not in LENGTH_PRESETS:
            raise ValueError(
                f"Unknown length '{self.length}'. "
                f"Choose from: {list(LENGTH_PRESETS)}"
            )
        if self.persona == "custom" and not self.custom_persona:
            raise ValueError(
                "persona='custom' requires a non-empty custom_persona string."
            )

    # ── Builder ────────────────────────────────────────────────────────────────

    def build_system_prompt(self) -> str:
        """
        Assembles the final system prompt from the three config axes + grounding.

        Structure (always 3-4 lines max):
            <persona line>
            <tone modifier>      ← omitted when "neutral"
            <length modifier>    ← omitted when "medium"
            <grounding — always present>
        """
        # Persona line
        persona_str = PERSONA_PRESETS[self.persona]
        if self.persona == "custom":
            persona_str = self.custom_persona  # type: ignore[assignment]

        # Assemble modifiers (skip empty strings)
        parts = [persona_str]
        tone_mod = TONE_PRESETS[self.tone]
        if tone_mod:
            parts.append(tone_mod)
        length_mod = LENGTH_PRESETS[self.length]
        if length_mod:
            parts.append(length_mod)
        parts.append(_GROUNDING)

        return "  ".join(parts)   # double-space separator keeps it one readable block

    # ── Serialisation (for API) ────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "persona":        self.persona,
            "tone":           self.tone,
            "length":         self.length,
            "custom_persona": self.custom_persona,
            "system_prompt":  self.build_system_prompt(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PersonaConfig":
        return cls(
            persona=data.get("persona", "sagan"),
            tone=data.get("tone", "neutral"),
            length=data.get("length", "medium"),
            custom_persona=data.get("custom_persona"),
        )

    # ── Catalogue helper (for the GET /api/persona endpoint) ──────────────────

    @staticmethod
    def catalogue() -> dict:
        """Return all available options — consumed by the frontend dropdowns."""
        return {
            "personas":  list(PERSONA_PRESETS.keys()),
            "tones":     list(TONE_PRESETS.keys()),
            "lengths":   list(LENGTH_PRESETS.keys()),
            "descriptions": {
                "personas":  {k: v.split("—")[-1].strip() if "—" in v else v[:60]
                              for k, v in PERSONA_PRESETS.items() if k != "custom"},
                "tones":     {k: (v[:60] if v else "No modifier") for k, v in TONE_PRESETS.items()},
                "lengths":   {k: (v[:60] if v else "Natural length") for k, v in LENGTH_PRESETS.items()},
            },
        }
