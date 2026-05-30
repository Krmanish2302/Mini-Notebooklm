"""
persona_config.py — Persona + response-style configuration for Chat mode.

Design: curated PRESETS are the primary UX — not a free-text field — because:
  1. Token efficiency  — presets are <15 words; free text balloons cost.
  2. Grounding safety  — every preset bakes in the strict source-only rule.
  3. Better UX         — dropdown + tone slider faster than custom prompts.
  4. Power-user escape — custom_persona field allows full override (advanced toggle).

Preset taxonomy:
  Persona  — *who* the assistant sounds like
  Tone     — *how* it sounds (energy, formality)
  Length   — *how much* it writes
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional


PERSONA_PRESETS: Dict[str, str] = {
    "sagan":      "You're Carl Sagan if he were a chill classmate — poetic, curious, full of wonder.",
    "professor":  "You're a clear-headed university professor — precise, structured, no fluff.",
    "eli5":       "You're explaining to a curious 12-year-old — simple words, fun analogies.",
    "analyst":    "You're a sharp research analyst — data-driven, concise, bullet-friendly.",
    "socratic":   "You're a Socratic tutor — answer questions with guiding questions, build insight step by step.",
    "journalist": "You're an investigative journalist — lead with the key finding, then unpack.",
    "custom":     "{custom}",
}

TONE_PRESETS: Dict[str, str] = {
    "casual":       "Keep it conversational and friendly.",
    "neutral":      "",
    "formal":       "Use formal, professional language.",
    "enthusiastic": "Be energetic and upbeat — make ideas feel exciting.",
    "stoic":        "Be calm, measured, and to the point.",
}

LENGTH_PRESETS: Dict[str, str] = {
    "short":   "Keep answers to 2-3 sentences max.",
    "medium":  "",
    "long":    "Give thorough, detailed answers with examples.",
    "bullets": "Always respond in bullet points.",
}

_GROUNDING = (
    "Use ONLY the sources below. "
    "Cite as [S1], [S2]… "
    "If it's not there, say: 'Not in my notes, bro.'"
)


@dataclass
class PersonaConfig:
    """
    Persona + style settings for Chat mode.

    persona        : key from PERSONA_PRESETS (default: "sagan")
    tone           : key from TONE_PRESETS    (default: "neutral")
    length         : key from LENGTH_PRESETS  (default: "medium")
    custom_persona : raw string — used only when persona=="custom"
    """
    persona:        str            = "sagan"
    tone:           str            = "neutral"
    length:         str            = "medium"
    custom_persona: Optional[str]  = None

    def __post_init__(self) -> None:
        if self.persona not in PERSONA_PRESETS:
            raise ValueError(f"Unknown persona '{self.persona}'. Choose from: {list(PERSONA_PRESETS)}")
        if self.tone not in TONE_PRESETS:
            raise ValueError(f"Unknown tone '{self.tone}'. Choose from: {list(TONE_PRESETS)}")
        if self.length not in LENGTH_PRESETS:
            raise ValueError(f"Unknown length '{self.length}'. Choose from: {list(LENGTH_PRESETS)}")
        if self.persona == "custom" and not self.custom_persona:
            raise ValueError("persona='custom' requires a non-empty custom_persona string.")

    def build_system_prompt(self) -> str:
        """
        Builds final system prompt: persona + tone modifier + length modifier + grounding.
        Grounding is ALWAYS appended regardless of persona override.
        """
        persona_str = (
            self.custom_persona
            if self.persona == "custom"
            else PERSONA_PRESETS[self.persona]
        )
        parts = [persona_str]
        if tone_mod := TONE_PRESETS[self.tone]:
            parts.append(tone_mod)
        if length_mod := LENGTH_PRESETS[self.length]:
            parts.append(length_mod)
        parts.append(_GROUNDING)
        return "  ".join(parts)

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

    @staticmethod
    def catalogue() -> dict:
        """All available options — consumed by frontend dropdowns."""
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