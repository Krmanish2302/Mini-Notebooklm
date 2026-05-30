"""
config.py — Centralised, type-validated application settings.

Priority order (highest → lowest):
  1. Environment variables          LLM_PROVIDER=openai python main.py
  2. .env file                      LLM_PROVIDER=openai in .env
  3. config.yaml overlay            llm.provider: openai in config.yaml
  4. Pydantic field defaults        provider="groq"

LangChain integration:
  - LangSmith tracing auto-enabled when LANGCHAIN_TRACING_V2=true + LANGCHAIN_API_KEY set.
  - LANGCHAIN_PROJECT controls the LangSmith project name.
  - All LANGCHAIN_* env vars are recognised automatically by LangChain SDKs;
    we surface them here as typed fields so they appear in settings dumps.

Usage:
    from src.core.config import get_settings
    s = get_settings()
    print(s.llm_provider)       # "groq"
    print(s.llm_model)          # "llama-3.1-70b-versatile"
    print(s.data_dir)           # "./data"

    # Override in tests:
    from unittest.mock import patch
    with patch("src.core.config._settings_cache", None):
        os.environ["LLM_PROVIDER"] = "openai"
        s = get_settings()
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

_CONFIG_YAML = os.getenv("CONFIG_PATH", "config.yaml")


# ── YAML overlay loader ───────────────────────────────────────────────────────

def _load_yaml_flat(path: str) -> dict:
    """
    Load config.yaml and flatten nested keys to UPPER_SNAKE env-var style.
    e.g. llm.provider → LLM_PROVIDER
    Only used as a fallback when env vars are absent.
    """
    if not Path(path).exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except Exception as exc:
        logger.warning("[Config] Failed to load %s: %s", path, exc)
        return {}

    flat: dict = {}

    def _flatten(d: dict, prefix: str = "") -> None:
        for k, v in d.items():
            key = (f"{prefix}_{k}" if prefix else k).upper()
            if isinstance(v, dict):
                _flatten(v, key)
            else:
                flat[key] = v

    _flatten(raw)
    return flat


# ── Settings model ────────────────────────────────────────────────────────────

class Settings(BaseSettings):
    """
    All application settings in one place.
    Field names match env var names (case-insensitive).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM ──────────────────────────────────────────────────────────────────
    llm_provider:    str   = Field("groq",                    description="groq|openai|gemini|ollama|anthropic")
    llm_model:       str   = Field("llama-3.1-70b-versatile", description="Model name for the chosen provider")
    llm_temperature: float = Field(0.7,  ge=0.0, le=2.0)
    llm_max_tokens:  int   = Field(1024, ge=64,  le=32768)

    # ── API Keys ─────────────────────────────────────────────────────────────
    groq_api_key:      Optional[str] = None
    openai_api_key:    Optional[str] = None
    google_api_key:    Optional[str] = None
    anthropic_api_key: Optional[str] = None

    # ── LangSmith / LangChain tracing ─────────────────────────────────────────
    langchain_tracing_v2:  bool          = False
    langchain_api_key:     Optional[str] = None
    langchain_project:     str           = "mini-notebooklm"
    langchain_endpoint:    str           = "https://api.smith.langchain.com"

    # ── Embeddings ────────────────────────────────────────────────────────────
    embedding_model:      str  = "all-MiniLM-L6-v2"
    embedding_batch_size: int  = Field(32, ge=1, le=512)
    embedding_dim:        int  = Field(384, ge=64, le=4096)

    # ── Storage ───────────────────────────────────────────────────────────────
    data_dir:        str = "./data"
    db_path:         str = "./data/metadata.db"
    vector_store_dir:str = "./data/vector_store"
    graph_path:      str = "./data/knowledge_graph/graph.pkl"

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retrieval_top_k:          int   = Field(10,  ge=1,   le=100)
    retrieval_rrf_k:          int   = Field(60,  ge=1,   le=200)
    retrieval_sim_threshold:  float = Field(0.3, ge=0.0, le=1.0)
    retrieval_bm25_weight:    float = Field(0.4, ge=0.0, le=1.0)
    retrieval_dense_weight:   float = Field(0.6, ge=0.0, le=1.0)

    # ── Ingestion ─────────────────────────────────────────────────────────────
    chunk_size:          int  = Field(512,  ge=64,   le=8192)
    chunk_overlap:       int  = Field(64,   ge=0,    le=1024)
    max_file_size_mb:    int  = Field(50,   ge=1,    le=500)
    allowed_extensions:  List[str] = Field(
        default_factory=lambda: ["pdf","txt","md","csv","png","jpg","jpeg","mp3","mp4","wav"]
    )

    # ── Evaluation ────────────────────────────────────────────────────────────
    eval_embedding_model:   str   = "all-MiniLM-L6-v2"
    eval_overlap_threshold: float = Field(0.25, ge=0.0, le=1.0)
    eval_enabled:           bool  = True

    # ── App ───────────────────────────────────────────────────────────────────
    app_name:    str  = "Mini NotebookLM"
    app_version: str  = "0.1.0"
    debug:       bool = False
    log_level:   str  = "INFO"

    # ── Validators ────────────────────────────────────────────────────────────

    @field_validator("llm_provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        allowed = {"groq", "openai", "gemini", "ollama", "anthropic"}
        if v.lower() not in allowed:
            raise ValueError(f"llm_provider must be one of {allowed}, got '{v}'")
        return v.lower()

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        u = v.upper()
        if u not in allowed:
            raise ValueError(f"log_level must be one of {allowed}, got '{v}'")
        return u

    @model_validator(mode="after")
    def _apply_yaml_overlay(self) -> "Settings":
        """
        Apply config.yaml values for any field still at its default.
        Env vars always win — yaml only fills gaps.
        """
        yaml_vals = _load_yaml_flat(_CONFIG_YAML)
        if not yaml_vals:
            return self
        for field_name in self.model_fields:
            env_key = field_name.upper()
            if env_key in os.environ:
                continue   # env var wins
            if env_key in yaml_vals:
                try:
                    object.__setattr__(self, field_name, yaml_vals[env_key])
                except Exception:
                    pass   # keep default if type coercion fails
        return self

    @model_validator(mode="after")
    def _configure_langsmith(self) -> "Settings":
        """Auto-set LangSmith env vars so LangChain SDK picks them up."""
        if self.langchain_tracing_v2 and self.langchain_api_key:
            os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
            os.environ.setdefault("LANGCHAIN_API_KEY",    self.langchain_api_key)
            os.environ.setdefault("LANGCHAIN_PROJECT",    self.langchain_project)
            os.environ.setdefault("LANGCHAIN_ENDPOINT",   self.langchain_endpoint)
        return self

    # ── Helpers ───────────────────────────────────────────────────────────────

    def ensure_dirs(self) -> None:
        """Create all data directories if they don't exist."""
        dirs = [
            self.data_dir,
            self.vector_store_dir,
            str(Path(self.graph_path).parent),
            str(Path(self.db_path).parent),
        ]
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)

    def active_api_key(self) -> Optional[str]:
        """Return the API key for the currently configured provider."""
        return {
            "groq":      self.groq_api_key,
            "openai":    self.openai_api_key,
            "gemini":    self.google_api_key,
            "anthropic": self.anthropic_api_key,
            "ollama":    None,
        }.get(self.llm_provider)

    def to_safe_dict(self) -> dict:
        """Return settings dict with all API keys redacted — safe for logging."""
        d = self.model_dump()
        for k in list(d):
            if "key" in k or "secret" in k or "password" in k:
                d[k] = "***" if d[k] else None
        return d


# ── Cached singleton ──────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the cached Settings singleton.
    Call get_settings.cache_clear() in tests to force re-instantiation.
    """
    s = Settings()
    logging.basicConfig(level=getattr(logging, s.log_level, logging.INFO))
    s.ensure_dirs()
    logger.info(
        "[Config] Loaded: provider=%s model=%s debug=%s tracing=%s",
        s.llm_provider, s.llm_model, s.debug, s.langchain_tracing_v2,
    )
    return s