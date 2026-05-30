"""
streamlit_app.py — Mini NotebookLM · Streamlit UI
Connects to the FastAPI backend at API_URL (default: http://localhost:8000/api).

Changes from original
---------------------
* RAGAS rendered via src.ui.render_ragas_panel (not raw st.json)
* render_grounding_badge shown inline after each assistant response
* show_ragas_sidebar added to sidebar bottom
* Persona controls added (preset dropdowns, not just temperature slider)
* st.session_state initialisation consolidated at top
* SSE error type displays st.error inline in chat bubble
* API_URL configurable via env var STREAMLIT_API_URL
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import requests
import streamlit as st

# ── Optional: use src.ui components if the package is importable ──────────
try:
    from src.ui import render_grounding_badge, render_ragas_panel, show_ragas_sidebar
    _HAS_UI = True
except ImportError:
    _HAS_UI = False

API_URL = os.getenv("STREAMLIT_API_URL", "http://localhost:8000/api")

st.set_page_config(
    page_title="Mini NotebookLM",
    layout="wide",
    page_icon="📓",
    initial_sidebar_state="expanded",
)

# ── Session state defaults ────────────────────────────────────────────────
_DEFAULTS: Dict[str, Any] = {
    "messages":     [],
    "current_mode": "chat",
    "temperature":  0.7,
    "top_p":        1.0,
    "persona":      "sagan",
    "tone":         "neutral",
    "length":       "medium",
}
for k, v in _DEFAULTS.items():
    st.session_state.setdefault(k, v)


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📓 Mini NotebookLM")
    st.caption("Local · open-source · RAG research assistant")

    # ── 1. LLM Config ─────────────────────────────────────────────────────
    st.subheader("⚙️ LLM Configuration")
    provider = st.selectbox("Provider", ["groq", "openai", "ollama", "gemini", "anthropic"])
    model    = st.text_input("Model", value="llama-3.1-70b-versatile")
    api_key  = st.text_input("API Key", type="password")
    if st.button("💾 Apply Config", use_container_width=True):
        try:
            r = requests.post(f"{API_URL}/config", json={"provider": provider, "model": model, "api_key": api_key}, timeout=10)
            st.success("Config updated ✓") if r.ok else st.error(r.text)
        except Exception as e:
            st.error(f"Backend unreachable: {e}")

    st.divider()

    # ── 2. Mode ───────────────────────────────────────────────────────────
    st.subheader("🧠 Pipeline Mode")
    mode = st.radio(
        "Mode",
        ["chat", "deep_research", "study"],
        format_func=lambda x: {"chat": "💬 Chat", "deep_research": "🔬 Deep Research", "study": "🎓 Study"}.get(x, x),
        label_visibility="collapsed",
    )
    if mode != st.session_state.current_mode:
        try:
            requests.post(f"{API_URL}/mode", json={"mode": mode}, timeout=5)
        except Exception:
            pass
        st.session_state.current_mode = mode
        st.session_state.messages = []
        st.rerun()

    st.divider()

    # ── 3. Persona ────────────────────────────────────────────────────────
    st.subheader("🎭 Persona & Style")
    persona = st.selectbox("Persona",  ["sagan", "professor", "eli5", "analyst", "socratic", "journalist"])
    tone    = st.selectbox("Tone",     ["neutral", "casual", "formal", "enthusiastic", "stoic"])
    length  = st.selectbox("Length",   ["medium", "short", "long", "bullets"])
    if st.button("Apply Persona", use_container_width=True):
        try:
            r = requests.post(f"{API_URL}/persona", json={"persona": persona, "tone": tone, "length": length}, timeout=5)
            if r.ok:
                st.session_state.persona = persona
                st.session_state.tone    = tone
                st.session_state.length  = length
                st.success("Persona updated ✓")
            else:
                st.error(r.text)
        except Exception as e:
            st.error(str(e))

    with st.expander("🔧 Inference Tuning"):
        st.session_state.temperature = st.slider("Temperature", 0.0, 2.0, st.session_state.temperature, 0.05)
        st.session_state.top_p       = st.slider("Top P",       0.0, 1.0, st.session_state.top_p,       0.05)

    st.divider()

    # ── 4. Knowledge Base ─────────────────────────────────────────────────
    st.subheader("📚 Knowledge Base")
    ingest_type = st.selectbox("Source Type", ["pdf", "website", "youtube", "csv", "text"])
    file_upload = None
    url_input   = None
    if ingest_type in ("pdf", "csv", "text"):
        file_upload = st.file_uploader(f"Upload {ingest_type.upper()}", type=["pdf","csv","txt"])
    else:
        url_input = st.text_input("Source URL", placeholder="https://...")

    if st.button("⬆️ Ingest Source", use_container_width=True):
        with st.spinner("Ingesting …"):
            try:
                if file_upload:
                    r = requests.post(
                        f"{API_URL}/ingest",
                        files={"file": (file_upload.name, file_upload.getvalue(), file_upload.type)},
                        data={"source_type": ingest_type}, timeout=120,
                    )
                elif url_input:
                    r = requests.post(f"{API_URL}/ingest", data={"url": url_input, "source_type": ingest_type}, timeout=120)
                else:
                    st.warning("Provide a file or URL.")
                    r = None
                if r:
                    st.success("Ingested ✓") if r.ok else st.error(r.text)
            except Exception as e:
                st.error(f"Backend unreachable: {e}")

    # ── 5. Active Sources ─────────────────────────────────────────────────
    st.subheader("📂 Active Sources")
    try:
        sr = requests.get(f"{API_URL}/sources", timeout=5)
        if sr.ok:
            sources = sr.json().get("sources", [])
            if not sources:
                st.caption("No sources yet.")
            for s in sources:
                c1, c2 = st.columns([5, 1])
                c1.caption(f"📄 {str(s.get('name', s.get('source_id', 'Unknown')))[:28]}")
                if c2.button("🗑", key=f"del_{s.get('id', s.get('source_id'))}"):
                    requests.delete(f"{API_URL}/sources/{s.get('id', s.get('source_id'))}", timeout=10)
                    st.rerun()
    except Exception:
        st.caption("Could not reach backend.")

    st.divider()

    # ── 6. RAGAS sidebar summary ──────────────────────────────────────────
    if _HAS_UI:
        try:
            hist = requests.get(f"{API_URL}/ragas/history?limit=1", timeout=3).json()
            last = (hist.get("history") or [None])[0]
            if last:
                show_ragas_sidebar(last)
        except Exception:
            pass


# ── Main chat area ─────────────────────────────────────────────────────────
mode_labels = {"chat": "💬 Chat", "deep_research": "🔬 Deep Research", "study": "🎓 Study"}
st.header(mode_labels.get(st.session_state.current_mode, st.session_state.current_mode))

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        meta = msg.get("metadata", {})
        ragas = msg.get("ragas")

        # Inline grounding badge
        if ragas and _HAS_UI:
            render_grounding_badge(ragas)
        elif ragas:
            st.caption(f"📎 Faithfulness {ragas.get('faithfulness', 0)*100:.0f}%  |  Grade: {ragas.get('grade','N/A')}")

        # Mode-specific metadata
        if meta:
            with st.expander("📋 Details"):
                if meta.get("sub_queries"):
                    st.markdown("**🔍 Sub-queries explored**")
                    for sq in meta["sub_queries"]: st.caption(f"• {sq}")
                if meta.get("summary_bullets"):
                    st.markdown("**📝 Key Takeaways**")
                    for b in meta["summary_bullets"]: st.write(f"• {b}")
                if meta.get("quiz_cards"):
                    st.markdown("**🎓 Flashcards**")
                    for card in meta["quiz_cards"]:
                        st.markdown(f"**Q:** {card.get('question')}")
                        st.markdown(f"**A:** {card.get('answer')} *(Difficulty: {card.get('difficulty','N/A')})*")
                        st.divider()
                if ragas and _HAS_UI:
                    render_ragas_panel(ragas, key=f"hist_{id(msg)}")


# ── Chat input ─────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask your documents anything …"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder  = st.empty()
        full_answer  = ""
        metadata: Dict[str, Any] = {}
        ragas_result: Optional[dict] = None

        payload = {
            "query":       prompt,
            "mode":        st.session_state.current_mode,
            "stream":      True,
            "temperature": st.session_state.temperature,
            "top_p":       st.session_state.top_p,
        }

        try:
            with requests.post(f"{API_URL}/query/stream", json=payload, stream=True, timeout=120) as r:
                r.raise_for_status()
                for raw_line in r.iter_lines():
                    if not raw_line:
                        continue
                    line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                    if not line.startswith("data: "):
                        continue
                    try:
                        event = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue

                    etype = event.get("type")
                    if etype == "token":
                        full_answer += event.get("content", "")
                        placeholder.markdown(full_answer + "▌")
                    elif etype == "metadata":
                        metadata = {k: v for k, v in event.items() if k != "type"}
                    elif etype == "ragas":
                        ragas_result = {k: v for k, v in event.items() if k != "type"}
                    elif etype == "error":
                        st.error(event.get("detail", "Unknown error"))
                    elif etype == "done":
                        break

            placeholder.markdown(full_answer)

            # Inline grounding badge
            if ragas_result and _HAS_UI:
                render_grounding_badge(ragas_result)

            # Mode-specific outputs
            if metadata:
                with st.expander("📋 Details", expanded=True):
                    if metadata.get("sub_queries"):
                        st.markdown("**🔍 Sub-queries explored**")
                        for sq in metadata["sub_queries"]: st.caption(f"• {sq}")
                    if metadata.get("summary_bullets"):
                        st.markdown("**📝 Key Takeaways**")
                        for b in metadata["summary_bullets"]: st.write(f"• {b}")
                    if metadata.get("quiz_cards"):
                        st.markdown("**🎓 Flashcards**")
                        for card in metadata["quiz_cards"]:
                            st.markdown(f"**Q:** {card.get('question')}")
                            st.markdown(f"**A:** {card.get('answer')} *(Difficulty: {card.get('difficulty','N/A')})*")
                            st.divider()
                    if ragas_result and _HAS_UI:
                        render_ragas_panel(ragas_result, key=f"live_{len(st.session_state.messages)}")

        except requests.exceptions.RequestException as e:
            st.error(f"Backend unreachable: {e}")

    st.session_state.messages.append({
        "role":     "assistant",
        "content":  full_answer,
        "metadata": metadata,
        "ragas":    ragas_result,
    })