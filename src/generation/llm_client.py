"""
llm_client.py  —  Flexible LLM client (Groq / Ollama / OpenAI / Gemini)

Fixes applied
-------------
BUG-C03  _build_model() previously fell off the end and returned None when
         the provider matched none of the elif branches.  Now raises ValueError
         explicitly so failures surface immediately at construction time.
BUG-Q01  Inner wrapper classes (_GroqWrapper, _OpenAIWrapper, _GeminiWrapper)
         were redefined from scratch on every _build_model() call, including
         every update_tuning() call.  Hoisted to module level; constructors
         capture the live client + params.
BUG-S03  API key validation: fail fast with a clear ValueError when neither the
         kwarg nor the expected env variable is set.  Without this, OpenAI /
         Gemini raise a vague AuthenticationError at first use.
BUG-Q02  Magic-number defaults (model name, temperature, max_tokens) are now
         documented as named constants at the top of the file.
"""
from __future__ import annotations

import os
import logging
from typing import Iterator, Optional

logger = logging.getLogger(__name__)

# ── Default constants (BUG-Q02) ───────────────────────────────────────────────
DEFAULT_PROVIDER     = "groq"
DEFAULT_MODEL        = "llama-3.1-70b-versatile"
DEFAULT_TEMPERATURE  = 0.7
DEFAULT_TOP_P        = 0.9
DEFAULT_MAX_TOKENS   = 1024


# ── Module-level wrapper classes (BUG-Q01) ────────────────────────────────────

class _GroqWrapper:
    def __init__(self, client, model_name: str, temperature: float, max_tokens: int):
        self._client = client
        self._model = model_name
        self._temperature = temperature
        self._max_tokens = max_tokens

    def invoke(self, prompt: str):
        class _R:
            def __init__(self, content): self.content = content
        comp = self._client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self._model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        return _R(comp.choices[0].message.content)

    def stream(self, prompt: str):
        class _C:
            def __init__(self, content): self.content = content
        for chunk in self._client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self._model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            stream=True,
        ):
            delta = chunk.choices[0].delta.content
            if delta is not None:
                yield _C(delta)


class _OpenAIWrapper:
    def __init__(self, client, model_name: str, temperature: float, max_tokens: int):
        self._client = client
        self._model = model_name
        self._temperature = temperature
        self._max_tokens = max_tokens

    def invoke(self, prompt: str):
        class _R:
            def __init__(self, content): self.content = content
        comp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        return _R(comp.choices[0].message.content)

    def stream(self, prompt: str):
        class _C:
            def __init__(self, content): self.content = content
        for chunk in self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            stream=True,
        ):
            delta = chunk.choices[0].delta.content
            if delta:
                yield _C(delta)


class _GeminiWrapper:
    def __init__(self, client, genai_mod, temperature: float, max_tokens: int):
        self._client = client
        self._genai = genai_mod
        self._temperature = temperature
        self._max_tokens = max_tokens

    def _gen_cfg(self):
        return self._genai.types.GenerationConfig(
            temperature=self._temperature,
            max_output_tokens=self._max_tokens,
        )

    def invoke(self, prompt: str):
        class _R:
            def __init__(self, content): self.content = content
        resp = self._client.generate_content(prompt, generation_config=self._gen_cfg())
        return _R(resp.text)

    def stream(self, prompt: str):
        class _C:
            def __init__(self, content): self.content = content
        for chunk in self._client.generate_content(
            prompt, generation_config=self._gen_cfg(), stream=True
        ):
            if chunk.text:
                yield _C(chunk.text)


# ── Main client ───────────────────────────────────────────────────────────────

class LLMClient:
    """
    Flexible LLM client supporting Groq, Ollama, OpenAI, and Gemini.

    All paths expose the same two-method contract:
        invoke(prompt: str) -> str
        stream(prompt: str) -> Iterator[str]
    """

    SUPPORTED_PROVIDERS = {"groq", "ollama", "openai", "gemini"}

    def __init__(
        self,
        provider: str = DEFAULT_PROVIDER,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        **kwargs,
    ):
        self.provider = provider.lower().strip()
        self.model_name = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self._api_key = api_key
        self._extra_kwargs = kwargs

        if self.provider not in self.SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Provider '{provider}' not supported. "
                f"Choose from: {sorted(self.SUPPORTED_PROVIDERS)}"
            )

        self.model = self._build_model()

    # ── Internal factory ──────────────────────────────────────────────────────

    def _build_model(self):
        """Build (or rebuild) the underlying model object from current params."""
        api_key = self._api_key

        if self.provider == "groq":
            try:
                from groq import Groq
            except ImportError:
                raise ImportError("pip install groq  # required for Groq provider")
            # BUG-S03: fail fast if key is missing
            resolved = api_key or os.environ.get("GROQ_API_KEY")
            if not resolved:
                raise ValueError(
                    "Groq API key not provided. Set GROQ_API_KEY env var or pass api_key=."
                )
            return _GroqWrapper(
                client=Groq(api_key=resolved),
                model_name=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

        if self.provider == "ollama":
            try:
                from langchain_community.chat_models import ChatOllama
            except ImportError:
                raise ImportError("pip install langchain-community  # required for Ollama")
            return ChatOllama(
                model=self.model_name,
                temperature=self.temperature,
                **self._extra_kwargs,
            )

        if self.provider == "openai":
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("pip install openai  # required for OpenAI provider")
            # BUG-S03: fail fast
            resolved = api_key or os.environ.get("OPENAI_API_KEY")
            if not resolved:
                raise ValueError(
                    "OpenAI API key not provided. Set OPENAI_API_KEY env var or pass api_key=."
                )
            return _OpenAIWrapper(
                client=OpenAI(api_key=resolved),
                model_name=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

        if self.provider == "gemini":
            try:
                import google.generativeai as genai
            except ImportError:
                raise ImportError("pip install google-generativeai  # required for Gemini")
            # BUG-S03: fail fast
            resolved = api_key or os.environ.get("GOOGLE_API_KEY")
            if not resolved:
                raise ValueError(
                    "Gemini API key not provided. Set GOOGLE_API_KEY env var or pass api_key=."
                )
            genai.configure(api_key=resolved)
            return _GeminiWrapper(
                client=genai.GenerativeModel(self.model_name),
                genai_mod=genai,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

        # BUG-C03: explicit guard — should be unreachable but makes failures obvious
        raise ValueError(f"_build_model: unhandled provider '{self.provider}'")

    # ── Public interface ──────────────────────────────────────────────────────

    def invoke(self, prompt: str) -> str:
        """Generate a full response (blocking). Always returns a plain str."""
        return self.model.invoke(prompt).content

    def stream(self, prompt: str) -> Iterator[str]:
        """Generate a response token by token. Yields plain str tokens."""
        for chunk in self.model.stream(prompt):
            yield chunk.content

    def update_tuning(self, **kwargs) -> None:
        """Update generation parameters at runtime and rebuild the model."""
        changed = False
        for key in ("temperature", "top_p", "max_tokens"):
            if key in kwargs:
                setattr(self, key, kwargs[key])
                changed = True
        if changed:
            self.model = self._build_model()
            logger.info(
                "LLMClient: tuning updated — temp=%.2f max_tokens=%d",
                self.temperature,
                self.max_tokens,
            )
