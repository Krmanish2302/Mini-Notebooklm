"""
persona_config.py — Persona + response-style configuration.

Preset taxonomy:
  Persona  — *who* the assistant sounds like
  Tone     — *how* it sounds
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
    "socratic":   "You're a Socratic tutor — answer with guiding questions, build insight step by step.",
    "journalist": "You're an investigative journalist — lead with the key finding, then unpack.",
    "custom":     "{custom}",
}

TONE_PRESETS: Dict[str, str] = {
    "casual":       "Keep it conversational and friendly.",
    "neutral":      "",
    "formal":       "Use formal, professional language.",
    "enthusiastic": "Be energetic and upbeat.",
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
    "Cite as [S1], [S2]\u2026 "
    "If it's not there, say: 'Not in my notes, bro.'"
)


@dataclass
class PersonaConfig:
    persona:        str           = "sagan"
    tone:           str           = "neutral"
    length:         str           = "medium"
    custom_persona: Optional[str] = None

    def __post_init__(self) -> None:
        if self.persona not in PERSONA_PRESETS:
            raise ValueError(f"Unknown persona '{self.persona}'. Choose: {list(PERSONA_PRESETS)}")
        if self.tone not in TONE_PRESETS:
            raise ValueError(f"Unknown tone '{self.tone}'. Choose: {list(TONE_PRESETS)}")
        if self.length not in LENGTH_PRESETS:
            raise ValueError(f"Unknown length '{self.length}'. Choose: {list(LENGTH_PRESETS)}")
        if self.persona == "custom" and not self.custom_persona:
            raise ValueError("persona='custom' requires a non-empty custom_persona string.")

    def build_system_prompt(self) -> str:
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
        return {
            "personas":  list(PERSONA_PRESETS.keys()),
            "tones":     list(TONE_PRESETS.keys()),
            "lengths":   list(LENGTH_PRESETS.keys()),
        }
