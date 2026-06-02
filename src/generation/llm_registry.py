"""
llm_registry.py

Single source of truth for LLM construction.
Uses LangChain-native chat models — no custom wrappers.

Supported providers (set LLM_PROVIDER env var):
  groq      — ChatGroq               (langchain-groq)
  openai    — ChatOpenAI             (langchain-openai)
  gemini    — ChatGoogleGenerativeAI (langchain-google-genai)
  ollama    — ChatOllama             (langchain-community)
  anthropic — ChatAnthropic          (langchain-anthropic)

Usage:
    from src.generation.llm_registry import LLMRegistry
    llm = LLMRegistry.get()                    # default from env
    llm = LLMRegistry.get(provider="openai")   # explicit
    response = llm.invoke("Hello")             # standard LangChain call
"""
from __future__ import annotations
import logging
import os
from functools import lru_cache
from typing import Optional

from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

DEFAULT_PROVIDER    = os.getenv("LLM_PROVIDER",     "groq")
DEFAULT_MODEL       = os.getenv("LLM_MODEL",        "llama-3.1-70b-versatile")
DEFAULT_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
DEFAULT_MAX_TOKENS  = int(os.getenv("LLM_MAX_TOKENS",    "1024"))


class LLMRegistry:
    """
    Factory + cache for LangChain chat models.
    Call LLMRegistry.get() to obtain the configured model.
    """

    @staticmethod
    @lru_cache(maxsize=8)
    def get(
        provider:    str   = DEFAULT_PROVIDER,
        model:       str   = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens:  int   = DEFAULT_MAX_TOKENS,
        api_key:     Optional[str] = None,
    ) -> BaseChatModel:
        """
        Return a cached LangChain chat model.
        Args are hashable so lru_cache works correctly.
        """
        p = provider.lower().strip()
        logger.info("[LLMRegistry] Building model: provider=%s model=%s", p, model)

        if p == "groq":
            from langchain_groq import ChatGroq
            key = api_key or os.environ.get("GROQ_API_KEY")
            if not key:
                raise ValueError("Set GROQ_API_KEY or pass api_key=")
            return ChatGroq(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                groq_api_key=key,
            )

        if p == "openai":
            from langchain_openai import ChatOpenAI
            base_url = os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL")
            key = api_key or os.environ.get("OPENAI_API_KEY")
            if not key and not base_url:
                raise ValueError("Set OPENAI_API_KEY, OPENAI_API_BASE, or pass api_key=")
            return ChatOpenAI(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                openai_api_key=key or "lm-studio",
                base_url=base_url,
            )

        if p == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            key = api_key or os.environ.get("GOOGLE_API_KEY")
            if not key:
                raise ValueError("Set GOOGLE_API_KEY or pass api_key=")
            return ChatGoogleGenerativeAI(
                model=model,
                temperature=temperature,
                max_output_tokens=max_tokens,
                google_api_key=key,
            )

        if p == "ollama":
            from langchain_community.chat_models import ChatOllama
            return ChatOllama(model=model, temperature=temperature)

        if p == "anthropic":
            from langchain_anthropic import ChatAnthropic
            key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise ValueError("Set ANTHROPIC_API_KEY or pass api_key=")
            return ChatAnthropic(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                anthropic_api_key=key,
            )

        raise ValueError(
            f"Unsupported provider '{provider}'. "
            "Choose: groq | openai | gemini | ollama | anthropic"
        )

    @staticmethod
    def get_streaming(
        provider:    str   = DEFAULT_PROVIDER,
        model:       str   = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens:  int   = DEFAULT_MAX_TOKENS,
    ) -> BaseChatModel:
        """Same as get() but streaming=True is set on construction."""
        return LLMRegistry.get(
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
