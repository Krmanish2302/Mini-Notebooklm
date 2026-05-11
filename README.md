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
- [API Reference](#api-reference)
- [Pipeline Modes](#pipeline-modes)
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

All data stays on your machine — no cloud vector DB, no third-party data sharing.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         React Frontend                          │
│                        (rag_ui.jsx / Vite)                      │
└──────────────────────────────┬──────────────────────────────────┘
                               │  HTTP / SSE
┌──────────────────────────────▼──────────────────────────────────┐
│                    FastAPI  (api.py)                            │
│  /api/ingest  /api/query  /api/query/stream  /api/evaluate      │
└──────┬──────────┬──────────────┬───────────────────────────────┘
       │          │              │
  ┌────▼────┐ ┌───▼──────────┐ ┌▼──────────────────┐
  │ Ingest  │ │ MasterPipeline│ │  Evaluation Layer  │
  │ Graph   │ │  (LangGraph)  │ │  (RAGAS metrics)   │
  └────┬────┘ └───┬──────────┘ └───────────────────┘
       │          │
       │   ┌──────▼───────────────────────────────────┐
       │   │           Pipeline Router                 │
       │   │  Chat | Deep Research | Study | Analyze   │
       │   └──────┬───────────────────────────────────┘
       │          │
  ┌────▼──────────▼──────────────────────────────────────────┐
  │                    Retrieval Layer                         │
  │  HybridRetriever → QueryRewriter → ContextBuilder         │
  │  SubQueryDecomposer → MultiQueryExpander                   │
  │  ContextualCompressor → Reranker → GraphRetriever          │
  └──────────────────────┬───────────────────────────────────┘
                         │
  ┌──────────────────────▼───────────────────────────────────┐
  │                    Storage Layer                          │
  │   FAISS (dense) + BM25 (sparse) + NetworkX Knowledge Graph│
  │   SQLite metadata  ·  File-based source registry          │
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
| 🗜️ **Context compression** | Jaccard dedup + token-budget management |
| 💬 **Chat mode** | Conversational RAG with full history |
| 🔬 **Deep Research mode** | Multi-hop retrieval + synthesis |
| 📚 **Study mode** | Quiz cards, summary bullets, learning path |
| 📊 **Evaluation** | RAGAS: faithfulness, relevancy, precision, recall |
| 🌐 **Streaming** | Server-Sent Events for real-time token streaming |
| 🔌 **Plugin system** | Configurable via `plugin_config.yaml` |
| 🐳 **Docker ready** | Single `docker-compose up` to start everything |

---

## Project Structure

```
Mini_NotebooLM/
├── api.py                          # FastAPI application & all REST endpoints
├── rag_ui.jsx                      # React frontend component
├── plugin_config.yaml              # Plugin & feature-flag configuration
├── pyproject.toml                  # Pytest & coverage config
├── requirements.txt                # All Python dependencies
├── Dockerfile                      # Production Docker image
├── docker-compose.yml              # Multi-service orchestration
├── .env.example                    # Environment variable template
│
├── src/
│   ├── master_pipeline.py          # Top-level pipeline orchestrator
│   ├── core/                       # Config, logging, shared utilities
│   ├── ingestion/                  # PDF, URL, YouTube loaders + chunkers
│   ├── retrieval/
│   │   ├── hybrid_retriever.py     # FAISS + BM25 + RRF fusion
│   │   ├── query_rewriter.py       # HyDE / step-back rewriting
│   │   ├── query_expander.py       # SubQueryDecomposer, MultiQueryExpander
│   │   ├── context_builder.py      # Dedup + token budget + source labels
│   │   ├── reranker.py             # Cross-encoder reranking
│   │   └── graph_retriever.py      # Knowledge graph traversal
│   ├── generation/
│   │   ├── response_generator.py   # LLM call + structured output parser
│   │   └── prompt_builder.py       # Prompt templates per mode
│   ├── pipelines/
│   │   ├── chat_pipeline.py        # Conversational RAG pipeline
│   │   ├── deep_research_pipeline.py # Multi-hop research pipeline
│   │   ├── study_pipeline.py       # Quiz + summary + learning path
│   │   ├── chat_graph.py           # LangGraph chat state machine
│   │   └── ingest_graph.py         # LangGraph ingestion state machine
│   ├── evaluation/                 # RAGAS evaluation harness
│   ├── graph/                      # Knowledge graph builder
│   ├── storage/                    # FAISS, SQLite, source registry
│   ├── agents/                     # Tool-calling agents
│   ├── chat_history/               # Conversation memory
│   └── ui/                         # Streamlit UI (optional)
│
├── frontend/                       # Vite + React project (if separated)
├── data/                           # Runtime data (gitignored)
│   ├── vector_store/               # FAISS index files
│   ├── sources.db                  # SQLite source metadata
│   └── graph/                      # Knowledge graph snapshots
│
├── tests/
│   ├── unit/                       # Pure unit tests
│   └── integration/                # FastAPI TestClient integration tests
│
└── .github/
    └── workflows/
        └── tests.yml               # CI pipeline
```

---

## Quick Start

### Prerequisites

- Python **3.11** or **3.12**
- `ffmpeg` (for audio/video processing): `sudo apt install ffmpeg` or `brew install ffmpeg`
- At least one LLM API key — see [Supported LLM Providers](#supported-llm-providers)

### Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/kumar2302github/Mini_NotebooLM.git
cd Mini_NotebooLM

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env — add your GROQ_API_KEY or OPENAI_API_KEY

# 5. Create data directories
mkdir -p data/vector_store data/graph

# 6. Start the API server
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`.
Interactive docs: `http://localhost:8000/docs`

#### Optional: Run the React frontend

```bash
# In a second terminal
cd frontend          # or project root if rag_ui.jsx is there
npm install
npm run dev
# Open http://localhost:5173
```

### Docker Setup

```bash
# 1. Copy and fill in your .env
cp .env.example .env

# 2. Build and start
docker compose up --build

# API  → http://localhost:8000
# Docs → http://localhost:8000/docs
```

To run in the background:
```bash
docker compose up -d --build
```

To stop:
```bash
docker compose down
```

---

## Configuration

All secrets and paths go in `.env` (copied from `.env.example`):

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | One of these | Groq LLM API key |
| `OPENAI_API_KEY` | One of these | OpenAI API key |
| `OLLAMA_BASE_URL` | One of these | Local Ollama base URL (default: `http://localhost:11434`) |
| `TAVILY_API_KEY` | Optional | Web search for deep research mode |
| `EMBEDDING_MODEL` | Optional | Sentence-transformers model name (default: `all-MiniLM-L6-v2`) |
| `FAISS_INDEX_PATH` | Optional | Path for FAISS index (default: `data/vector_store`) |
| `SQLITE_DB_PATH` | Optional | SQLite source metadata DB (default: `data/sources.db`) |
| `GRAPH_STORAGE_PATH` | Optional | Knowledge graph directory (default: `data/graph`) |
| `LOG_LEVEL` | Optional | `DEBUG` / `INFO` / `WARNING` / `ERROR` (default: `INFO`) |

Feature flags and plugin settings are managed in `plugin_config.yaml`.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Health check |
| `GET` | `/api/stats` | Index and source statistics |
| `POST` | `/api/config` | Update runtime configuration |
| `POST` | `/api/mode` | Switch pipeline mode (`chat` / `deep_research` / `study` / `analyze`) |
| `POST` | `/api/ingest` | Ingest a source (PDF file, URL, or YouTube link) |
| `GET` | `/api/sources` | List all ingested sources |
| `DELETE` | `/api/sources/{id}` | Remove a source |
| `POST` | `/api/query` | Query (non-streaming) |
| `POST` | `/api/query/stream` | Query with SSE streaming |
| `POST` | `/api/analyze` | Analyze a document without querying |
| `POST` | `/api/evaluate` | Run RAGAS evaluation on a QA pair |
| `GET` | `/api/ragas/history` | Get RAGAS evaluation history |

Full interactive documentation available at `/docs` (Swagger UI) and `/redoc`.

---

## Pipeline Modes

### 💬 Chat
Standard conversational RAG. Uses `HybridRetriever` + `QueryRewriter` + `ContextBuilder` + LLM. Maintains full conversation history across turns.

### 🔬 Deep Research
Breaks the question into sub-queries via `SubQueryDecomposer`, runs parallel retrieval for each, deduplicates, reranks, and synthesises a long-form answer. Returns `sub_queries` in the response for transparency.

### 📚 Study
Builds on Deep Research and additionally generates:
- **Quiz cards** — question / answer / difficulty triplets
- **Summary bullets** — 3–5 key takeaways
- **Learning path** — concept sequence from the knowledge graph

### 🔍 Analyze
One-shot document analysis without persistent chat history. Good for summarisation and structured extraction.

---

## Running Tests

```bash
# Install dependencies (includes test packages)
pip install -r requirements.txt

# Run all tests
pytest

# Unit tests only
pytest tests/unit -v

# Integration tests only
pytest tests/integration -v

# With coverage report
pytest --cov=src --cov=api --cov-report=term-missing
```

The CI pipeline runs automatically on every push and pull request via GitHub Actions.

---

## Supported LLM Providers

| Provider | Key variable | Notes |
|---|---|---|
| **Groq** | `GROQ_API_KEY` | Fast inference, free tier available |
| **OpenAI** | `OPENAI_API_KEY` | GPT-4o, GPT-4-turbo, etc. |
| **Ollama** | `OLLAMA_BASE_URL` | 100% local — no key needed, install Ollama separately |

You only need **one** provider configured. The app auto-selects based on which key is present.

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes and add tests
4. Ensure all tests pass: `pytest`
5. Open a pull request

---

<div align="center">
  <sub>Built with ❤️ using FastAPI · LangChain · LangGraph · FAISS · sentence-transformers</sub>
</div>
