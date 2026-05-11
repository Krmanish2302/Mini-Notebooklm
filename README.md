<!-- Banner -->
<div align="center">

<h1>📓 Mini NotebookLM</h1>

<p>
  <b>A local, open-source RAG research assistant</b> — ingest PDFs, web pages, and YouTube videos,
  then chat, deep-research, or study them with any LLM you choose.
</p>

[![Tests](https://github.com/kumar2302github/Mini_NotebooLM/actions/workflows/tests.yml/badge.svg)](https://github.com/kumar2302github/Mini_NotebooLM/actions/workflows/tests.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-009688.svg)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.3%2B-121212.svg)](https://langchain.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-FF6B35.svg)](https://langchain-ai.github.io/langgraph/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
  - [Local Setup](#local-setup)
  - [Docker Setup](#docker-setup)
- [Configuration](#configuration)
  - [Environment Variables](#environment-variables)
  - [Plugin Config YAML](#plugin-config-yaml)
- [API Reference](#api-reference)
- [Pipeline Modes](#pipeline-modes)
- [Plugin System](#plugin-system)
- [Running Tests](#running-tests)
- [Supported LLM Providers](#supported-llm-providers)
- [Contributing](#contributing)

---

## Overview

Mini NotebookLM is a **self-hosted, privacy-first research assistant** inspired by Google's NotebookLM.
It lets you:

- **Ingest** PDFs, web URLs, and YouTube video transcripts into a local vector store
- **Chat** with your documents using any LLM (Groq, OpenAI, or local Ollama)
- **Deep Research** complex questions via multi-hop sub-query decomposition
- **Study Mode** — auto-generates quiz cards, summary bullets, and a learning path
- **Evaluate** answers automatically using RAGAS metrics
- **Extend** any subsystem (LLM, embedder, vector store, document processor) via plugins

All data stays on your machine — no cloud vector DB, no third-party data sharing.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        React Frontend (rag_ui.jsx)                │
└──────────────────────────────┬──────────────────────────────────┘
                               │  HTTP / SSE
┌──────────────────────────────▼──────────────────────────────────┐
│                    FastAPI (api.py)                               │
│  /ingest  /query  /query/stream  /mode  /evaluate  /health        │
└──────┬──────────┬─────────────┬─────────────────────────────┘
       │          │             │
  ┌────▼────┐  ┌───▼──────────┐  ┌▼─────────────────┐
  │Ingest  │  │  MasterPipeline  │  │ Evaluation (RAGAS)  │
  │Graph   │  │  (LangGraph)     │  └───────────────────┘
  └────┬────┘  └───┬──────────┘
       │          │
       │  ┌──────▼───────────────────────────────────┐
       │  │  Pipeline Router (plugin_config.yaml drives this) │
       │  │  Chat | Deep Research | Study | Analyze          │
       │  └──────┬───────────────────────────────────┘
       │         │
  ┌────▼─────────▼───────────────────────────────────────────┐
  │              Retrieval Layer                                │
  │  HybridRetriever → QueryRewriter → ContextBuilder          │
  │  SubQueryDecomposer → MultiQueryExpander                    │
  │  ContextualCompressor → Reranker → GraphRetriever           │
  └──────────────────────┬───────────────────────────────────┘
                         │
  ┌──────────────────────▼───────────────────────────────────┐
  │              Storage Layer                                  │
  │  FAISS (dense) + BM25 (sparse) + NetworkX Knowledge Graph   │
  │  SQLite metadata  ·  File-based source registry             │
  └──────────────────────────────────────────────────────────┘
```

---

## Features

| Feature | Details |
|---|---|
| 🗂️ **Multi-source ingestion** | PDF, web URL (Trafilatura), YouTube transcript |
| 🔍 **Hybrid retrieval** | Dense (FAISS + sentence-transformers) + sparse (BM25) with RRF fusion |
| 🔁 **Query rewriting** | HyDE + step-back prompting |
| 🧩 **Sub-query decomposition** | Complex questions split into focused sub-queries |
| 🗜️ **Context compression** | Jaccard dedup + token-budget management (3000 token ceiling) |
| 💬 **Chat mode** | Conversational RAG with full history |
| 🔬 **Deep Research mode** | Multi-hop retrieval + synthesis across N sub-queries |
| 📚 **Study mode** | Quiz cards, summary bullets, learning path |
| 📊 **Evaluation** | RAGAS: faithfulness, relevancy, precision, recall |
| 🌐 **Streaming** | Server-Sent Events for real-time token output |
| 🔌 **Plugin system** | Hot-swap LLMs, embedders, vector stores, and processors via `plugin_config.yaml` |
| 🐳 **Docker ready** | One `docker compose up --build` to run everything |

---

## Project Structure

```
Mini_NotebooLM/
├── api.py                          # FastAPI app — all REST endpoints
├── rag_ui.jsx                      # React frontend component
├── plugin_config.yaml              # ★ Central config — model/provider single source of truth
├── pyproject.toml                  # Pytest & coverage config
├── requirements.txt                # All Python dependencies
├── Dockerfile                      # Production multi-stage Docker image
├── docker-compose.yml              # Multi-service orchestration
├── .env.example                    # Environment variable template
│
├── src/
│   ├── master_pipeline.py          # Top-level pipeline orchestrator
│   ├── core/                       # Config loader, LLM base classes, logging
│   ├── ingestion/                  # PDF, URL, YouTube loaders + chunkers
│   ├── retrieval/
│   │   ├── hybrid_retriever.py     # FAISS + BM25 + RRF fusion
│   │   ├── query_rewriter.py       # HyDE / step-back rewriting
│   │   ├── query_expander.py       # SubQueryDecomposer, MultiQueryExpander
│   │   ├── context_builder.py      # Dedup + token budget + citation labels
│   │   ├── reranker.py             # Cross-encoder reranking
│   │   └── graph_retriever.py      # Knowledge-graph traversal
│   ├── generation/
│   │   ├── response_generator.py   # LLM call + structured output parser
│   │   └── prompt_builder.py       # Prompt templates per pipeline mode
│   ├── pipelines/
│   │   ├── chat_pipeline.py
│   │   ├── deep_research_pipeline.py
│   │   ├── study_pipeline.py
│   │   ├── chat_graph.py           # LangGraph chat state machine
│   │   └── ingest_graph.py         # LangGraph ingestion state machine
│   ├── evaluation/                 # RAGAS evaluation harness
│   ├── graph/                      # Knowledge graph builder (NetworkX)
│   ├── storage/                    # FAISS index, SQLite, source registry
│   ├── agents/                     # Tool-calling agents
│   ├── chat_history/               # Conversation memory
│   └── ui/                         # Streamlit UI (optional)
│
├── frontend/                       # Vite + React project
├── data/                           # Runtime data (gitignored)
│   ├── vector_store/               # FAISS index files
│   ├── sources.db                  # SQLite source metadata
│   └── graph/                      # Knowledge graph snapshots
│
├── tests/
│   ├── unit/                       # Pure unit tests
│   └── integration/                # FastAPI TestClient integration tests
│
└── .github/workflows/tests.yml     # CI pipeline (unit → integration → docker build)
```

---

## Quick Start

### Prerequisites

- Python **3.11** or **3.12**
- `ffmpeg`: `sudo apt install ffmpeg` / `brew install ffmpeg`
- At least one LLM API key — see [Supported LLM Providers](#supported-llm-providers)

### Local Setup

```bash
# 1. Clone
git clone https://github.com/kumar2302github/Mini_NotebooLM.git
cd Mini_NotebooLM

# 2. Virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Configure
cp .env.example .env
# Edit .env — add GROQ_API_KEY or OPENAI_API_KEY

# 5. Create data directories
mkdir -p data/vector_store data/graph

# 6. Start API
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

- API: `http://localhost:8000`
- Swagger docs: `http://localhost:8000/docs`

#### Optional: React frontend

```bash
cd frontend
npm install && npm run dev
# Open http://localhost:5173
```

### Docker Setup

```bash
cp .env.example .env          # fill in your API keys
docker compose up --build

# API  → http://localhost:8000
# Docs → http://localhost:8000/docs
```

Background mode: `docker compose up -d --build`
Stop: `docker compose down`

---

## Configuration

### Environment Variables

All secrets go in `.env` (copied from `.env.example`):

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | One of these | Groq LLM API key |
| `OPENAI_API_KEY` | One of these | OpenAI API key |
| `OLLAMA_BASE_URL` | One of these | Local Ollama URL (default: `http://localhost:11434`) |
| `TAVILY_API_KEY` | Optional | Web search for Deep Research mode |
| `EMBEDDING_MODEL` | Optional | Sentence-transformers model (default: `all-MiniLM-L6-v2`) |
| `FAISS_INDEX_PATH` | Optional | FAISS index directory (default: `data/vector_store`) |
| `SQLITE_DB_PATH` | Optional | SQLite metadata DB (default: `data/sources.db`) |
| `GRAPH_STORAGE_PATH` | Optional | Knowledge graph directory (default: `data/graph`) |
| `LOG_LEVEL` | Optional | `DEBUG` / `INFO` / `WARNING` / `ERROR` (default: `INFO`) |

### Plugin Config YAML

`plugin_config.yaml` is the **single source of truth** for models, providers, and feature settings.
Change these two lines to switch provider and model everywhere at once:

```yaml
global:
  active_provider: groq                         # groq | openai | ollama
  active_chat_model: llama-3.3-70b-versatile    # change once → reflects in all pipelines
```

All pipeline sections (`chat`, `deep_research`, `study`, `analyze`, `evaluation`) inherit from
`global.active_provider` and `global.active_chat_model` unless explicitly overridden.

To enable a plugin (e.g., Pinecone vector store):
```yaml
vector_stores:
  plugins:
    - id: pinecone
      enabled: true   # flip this to true
      config:
        api_key_env: PINECONE_API_KEY
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Health check |
| `GET` | `/api/stats` | Index and source statistics |
| `POST` | `/api/config` | Update runtime config |
| `POST` | `/api/mode` | Switch pipeline mode (`chat` / `deep_research` / `study` / `analyze`) |
| `POST` | `/api/ingest` | Ingest a source (PDF file, URL, YouTube link) |
| `GET` | `/api/sources` | List all ingested sources |
| `DELETE` | `/api/sources/{id}` | Remove a source |
| `POST` | `/api/query` | Query (non-streaming) |
| `POST` | `/api/query/stream` | Query with SSE streaming |
| `POST` | `/api/analyze` | Analyze a document |
| `POST` | `/api/evaluate` | Run RAGAS evaluation |
| `GET` | `/api/ragas/history` | RAGAS evaluation history |

Full interactive docs at `/docs` (Swagger) and `/redoc`.

---

## Pipeline Modes

### 💬 Chat
Conversational RAG: `HybridRetriever` → `QueryRewriter` → `ContextBuilder` → LLM. Maintains full conversation history. Supports streaming via SSE.

### 🔬 Deep Research
Decomposes the question into N sub-queries via `SubQueryDecomposer`, runs parallel retrieval, deduplicates, reranks, and synthesises a long-form answer. Returns `sub_queries` in the response.

### 📚 Study
Builds on Deep Research and additionally generates:
- **Quiz cards** — Q / A / difficulty triplets
- **Summary bullets** — 3–5 key takeaways
- **Learning path** — concept sequence from the knowledge graph

### 🔍 Analyze
One-shot structured extraction without chat history. Best for summarisation and document analysis.

---

## Plugin System

Every subsystem is pluggable via `plugin_config.yaml`:

| Subsystem | Built-in | Plugin Slots |
|---|---|---|
| LLM Providers | Groq, OpenAI, Ollama | Claude, Azure OpenAI, HuggingFace Local |
| Embeddings | sentence-transformers, OpenAI | Cohere, Jina AI |
| Vector Stores | FAISS | Pinecone, Qdrant, Weaviate, Milvus |
| Document Processors | PDF, Web, YouTube, TXT, CSV | DOCX, PPTX, EPUB |
| Chunking | Recursive, Semantic, Hierarchical | Sentence-level |
| Retrieval | Hybrid, Semantic, BM25 | Cross-encoder reranker |
| Evaluation | RAGAS, ROUGE | DeepEval |
| Knowledge Graph | NetworkX | Neo4j |
| Storage | SQLite | PostgreSQL, MongoDB |

To enable any plugin, set `enabled: true` in the relevant section of `plugin_config.yaml`.

---

## Running Tests

```bash
# Install deps
pip install -r requirements.txt

# All tests
pytest

# Unit only
pytest tests/unit -v

# Integration only
pytest tests/integration -v

# With coverage
pytest --cov=src --cov=api --cov-report=term-missing
```

CI runs automatically on every push/PR via GitHub Actions (unit → integration → Docker build).

---

## Supported LLM Providers

| Provider | Key Variable | Notes |
|---|---|---|
| **Groq** | `GROQ_API_KEY` | Fast inference, free tier available |
| **OpenAI** | `OPENAI_API_KEY` | GPT-4.1, GPT-4.1-mini etc. |
| **Ollama** | `OLLAMA_BASE_URL` | 100% local — no key needed |
| **Claude** | `ANTHROPIC_API_KEY` | Plugin (set `enabled: true` in `plugin_config.yaml`) |
| **Azure OpenAI** | `AZURE_OPENAI_KEY` | Plugin (set `enabled: true` in `plugin_config.yaml`) |

You only need **one** provider configured. The app selects based on which key is present and `global.active_provider` in `plugin_config.yaml`.

---

## Contributing

1. Fork the repo
2. Create a branch: `git checkout -b feature/my-feature`
3. Add changes + tests
4. Verify: `pytest`
5. Open a pull request

---

<div align="center">
  <sub>Built with ❤️ using FastAPI · LangChain · LangGraph · FAISS · sentence-transformers</sub>
</div>
