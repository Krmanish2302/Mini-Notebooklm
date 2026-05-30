"""
streamlit_app.py — Mini NotebookLM · Streamlit UI (v2)

What changed vs v1
------------------
* Source checkboxes — each ingested source has a toggle;
  only checked sources are sent as source_ids in query payload
* 5 source-type tabs in Knowledge Base — PDF / YouTube / Website / Text / Image
* PDF strategy picker — dropdown populated after upload
* Paste tab — text area for raw text ingestion
* Progress bar during ingest via st.progress + st.status
* Backend status badge (green/red) at sidebar top
* Citations expander rendered cleanly per message
* source_ids forwarded to /api/query/stream
* API_URL configurable via STREAMLIT_API_URL env var (unchanged)
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import requests
import streamlit as st

# ── Optional src.ui components ───────────────────────────────────────────
try:
    from src.ui import render_grounding_badge, render_ragas_panel, show_ragas_sidebar
    _HAS_UI = True
except ImportError:
    _HAS_UI = False

API_URL = os.getenv("STREAMLIT_API_URL", "http://localhost:8000/api")

PDF_STRATEGIES = [
    "paragraph_based", "sentence_based", "fixed_size",
    "semantic", "recursive", "page_based",
]

st.set_page_config(
    page_title="Mini NotebookLM",
    layout="wide",
    page_icon="📓",
    initial_sidebar_state="expanded",
)

# ── Session state defaults ────────────────────────────────────────────
_DEFAULTS: Dict[str, Any] = {
    "messages":        [],
    "current_mode":    "chat",
    "temperature":     0.7,
    "top_p":           1.0,
    "persona":         "sagan",
    "tone":            "neutral",
    "length":          "medium",
    "active_sources":  {},   # {source_id: True/False} — checkbox states
}
for k, v in _DEFAULTS.items():
    st.session_state.setdefault(k, v)


# ── Helpers ────────────────────────────────────────────────────────────────

def _get(path: str, **kwargs) -> Optional[requests.Response]:
    try:
        return requests.get(f"{API_URL}{path}", timeout=5, **kwargs)
    except Exception:
        return None

def _post(path: str, **kwargs) -> Optional[requests.Response]:
    try:
        return requests.post(f"{API_URL}{path}", timeout=30, **kwargs)
    except Exception:
        return None

def _delete(path: str) -> Optional[requests.Response]:
    try:
        return requests.delete(f"{API_URL}{path}", timeout=10)
    except Exception:
        return None

def _backend_ok() -> bool:
    r = _get("/health")
    return r is not None and r.ok

def _fetch_sources() -> List[dict]:
    r = _get("/sources")
    if r and r.ok:
        return r.json().get("sources", [])
    return []

def _active_source_ids() -> List[str]:
    """Return source_ids where checkbox is True."""
    return [
        sid for sid, checked
        in st.session_state.active_sources.items()
        if checked
    ]

def _ingest_file(
    file_obj,
    source_type: str,
    source_id:   str,
    strategy:    str = "paragraph_based",
) -> Optional[requests.Response]:
    return requests.post(
        f"{API_URL}/ingest",
        files={"file": (file_obj.name, file_obj.getvalue(), file_obj.type)},
        data={"source_type": source_type, "source_id": source_id, "chunking_strategy": strategy},
        timeout=180,
    )

def _ingest_url(
    url:         str,
    source_type: str,
    source_id:   str,
) -> Optional[requests.Response]:
    return requests.post(
        f"{API_URL}/ingest",
        data={"url": url, "source_type": source_type, "source_id": source_id},
        timeout=180,
    )

def _ingest_text(
    text:      str,
    source_id: str,
) -> Optional[requests.Response]:
    return requests.post(
        f"{API_URL}/ingest",
        data={"content": text, "source_type": "text", "source_id": source_id},
        timeout=120,
    )


# ── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    # ── Header + status badge ────────────────────────────────────────
    col_title, col_badge = st.columns([4, 1])
    col_title.title("📓 Mini NotebookLM")
    if _backend_ok():
        col_badge.markdown("<div style='margin-top:28px;font-size:22px'>🟢</div>", unsafe_allow_html=True)
    else:
        col_badge.markdown("<div style='margin-top:28px;font-size:22px'>🔴</div>", unsafe_allow_html=True)
    st.caption("Local · open-source · RAG research assistant")

    # ── 1. LLM Config ──────────────────────────────────────────────
    st.subheader("⚙️ LLM Configuration")
    provider = st.selectbox("Provider", ["groq", "openai", "ollama", "gemini", "anthropic"])
    model    = st.text_input("Model", value="llama-3.1-70b-versatile")
    api_key  = st.text_input("API Key", type="password")
    if st.button("💾 Apply Config", use_container_width=True):
        r = _post("/config", json={"provider": provider, "model": model, "api_key": api_key})
        if r and r.ok:
            st.success("Config updated ✓")
        elif r:
            st.error(r.text)
        else:
            st.error("Backend unreachable")

    st.divider()

    # ── 2. Pipeline Mode ─────────────────────────────────────────────
    st.subheader("🧠 Pipeline Mode")
    mode = st.radio(
        "Mode",
        ["chat", "deep_research", "study"],
        format_func=lambda x: {
            "chat":          "💬 Chat",
            "deep_research": "🔬 Deep Research",
            "study":         "🎓 Study",
        }.get(x, x),
        label_visibility="collapsed",
    )
    if mode != st.session_state.current_mode:
        _post("/mode", json={"mode": mode})
        st.session_state.current_mode = mode
        st.session_state.messages = []
        st.rerun()

    st.divider()

    # ── 3. Persona & Style ──────────────────────────────────────────
    st.subheader("🎭 Persona & Style")
    persona = st.selectbox("Persona", ["sagan", "professor", "eli5", "analyst", "socratic", "journalist"])
    tone    = st.selectbox("Tone",    ["neutral", "casual", "formal", "enthusiastic", "stoic"])
    length  = st.selectbox("Length",  ["medium", "short", "long", "bullets"])
    if st.button("Apply Persona", use_container_width=True):
        r = _post("/persona", json={"persona": persona, "tone": tone, "length": length})
        if r and r.ok:
            st.session_state.persona = persona
            st.session_state.tone    = tone
            st.session_state.length  = length
            st.success("Persona updated ✓")
        elif r:
            st.error(r.text)

    with st.expander("🔧 Inference Tuning"):
        st.session_state.temperature = st.slider("Temperature", 0.0, 2.0, st.session_state.temperature, 0.05)
        st.session_state.top_p       = st.slider("Top P",       0.0, 1.0, st.session_state.top_p,       0.05)

    st.divider()

    # ── 4. Knowledge Base — 5 tabs ────────────────────────────────────
    st.subheader("📚 Knowledge Base")
    tab_pdf, tab_yt, tab_web, tab_txt, tab_img = st.tabs([
        "📄 PDF", "🎥 YouTube", "🌐 Website", "📝 Text", "🖼️ Image"
    ])

    # ─ PDF ───────────────────────────────────────────────────────────────
    with tab_pdf:
        pdf_file = st.file_uploader("Upload PDF", type=["pdf"], key="pdf_upload")
        if pdf_file:
            pdf_strategy = st.selectbox(
                "Chunking strategy",
                PDF_STRATEGIES,
                index=0,
                key="pdf_strategy",
                help="paragraph_based works well for most documents",
            )
            pdf_sid = st.text_input(
                "Source ID (optional)", key="pdf_sid",
                placeholder="e.g. lecture_notes_week1",
            )
            if st.button("⬆️ Ingest PDF", use_container_width=True, key="btn_pdf"):
                sid = pdf_sid.strip() or pdf_file.name.replace(" ", "_")[:20]
                with st.status(f"Ingesting {pdf_file.name} …", expanded=True) as status:
                    st.write("Uploading file …")
                    bar = st.progress(20)
                    try:
                        r = _ingest_file(pdf_file, "pdf", sid, pdf_strategy)
                        bar.progress(90)
                        if r and r.ok:
                            st.session_state.active_sources[sid] = True
                            bar.progress(100)
                            status.update(label="✅ Ingested!", state="complete")
                            st.rerun()
                        else:
                            status.update(label="❌ Failed", state="error")
                            st.error(r.text if r else "Backend unreachable")
                    except Exception as e:
                        status.update(label="❌ Error", state="error")
                        st.error(str(e))

    # ─ YouTube ──────────────────────────────────────────────────────────
    with tab_yt:
        yt_url = st.text_input("YouTube URL", placeholder="https://youtu.be/...", key="yt_url")
        yt_sid = st.text_input("Source ID (optional)", key="yt_sid", placeholder="e.g. talk_on_rag")
        if st.button("⬆️ Ingest YouTube", use_container_width=True, key="btn_yt"):
            if not yt_url.strip():
                st.warning("Provide a YouTube URL.")
            else:
                sid = yt_sid.strip() or "yt_" + yt_url.split("v=")[-1][:8]
                with st.status("Fetching transcript …", expanded=True) as status:
                    st.write("Downloading & chunking …")
                    bar = st.progress(30)
                    try:
                        r = _ingest_url(yt_url, "youtube", sid)
                        bar.progress(90)
                        if r and r.ok:
                            st.session_state.active_sources[sid] = True
                            bar.progress(100)
                            status.update(label="✅ Ingested!", state="complete")
                            st.rerun()
                        else:
                            status.update(label="❌ Failed", state="error")
                            st.error(r.text if r else "Backend unreachable")
                    except Exception as e:
                        status.update(label="❌ Error", state="error")
                        st.error(str(e))

    # ─ Website ───────────────────────────────────────────────────────────
    with tab_web:
        web_url = st.text_input("Website URL", placeholder="https://example.com", key="web_url")
        web_sid = st.text_input("Source ID (optional)", key="web_sid", placeholder="e.g. openai_blog")
        if st.button("⬆️ Ingest Website", use_container_width=True, key="btn_web"):
            if not web_url.strip():
                st.warning("Provide a URL.")
            else:
                sid = web_sid.strip() or "web_" + web_url.split("//")[-1][:16].replace("/", "_")
                with st.status("Crawling page …", expanded=True) as status:
                    bar = st.progress(30)
                    try:
                        r = _ingest_url(web_url, "website", sid)
                        bar.progress(90)
                        if r and r.ok:
                            st.session_state.active_sources[sid] = True
                            bar.progress(100)
                            status.update(label="✅ Ingested!", state="complete")
                            st.rerun()
                        else:
                            status.update(label="❌ Failed", state="error")
                            st.error(r.text if r else "Backend unreachable")
                    except Exception as e:
                        status.update(label="❌ Error", state="error")
                        st.error(str(e))

    # ─ Text / Paste ──────────────────────────────────────────────────────
    with tab_txt:
        txt_option = st.radio("Input", ["Upload file", "Paste text"], horizontal=True, key="txt_option")
        txt_sid = st.text_input("Source ID (optional)", key="txt_sid", placeholder="e.g. notes_ch3")

        if txt_option == "Upload file":
            txt_file = st.file_uploader("Upload .txt / .md / .csv", type=["txt", "md", "csv"], key="txt_upload")
            if txt_file and st.button("⬆️ Ingest File", use_container_width=True, key="btn_txt_file"):
                sid = txt_sid.strip() or txt_file.name.replace(" ", "_")[:20]
                with st.status("Processing …", expanded=True) as status:
                    bar = st.progress(20)
                    try:
                        r = _ingest_file(txt_file, "text", sid)
                        bar.progress(90)
                        if r and r.ok:
                            st.session_state.active_sources[sid] = True
                            bar.progress(100)
                            status.update(label="✅ Ingested!", state="complete")
                            st.rerun()
                        else:
                            status.update(label="❌ Failed", state="error")
                            st.error(r.text if r else "Backend unreachable")
                    except Exception as e:
                        status.update(label="❌ Error", state="error")
                        st.error(str(e))
        else:
            pasted = st.text_area("Paste your text here", height=150, key="txt_paste")
            if st.button("⬆️ Ingest Text", use_container_width=True, key="btn_txt_paste"):
                if not pasted.strip():
                    st.warning("Nothing to ingest.")
                else:
                    sid = txt_sid.strip() or "paste_" + str(len(pasted))[:6]
                    with st.status("Processing …", expanded=True) as status:
                        bar = st.progress(30)
                        try:
                            r = _ingest_text(pasted, sid)
                            bar.progress(90)
                            if r and r.ok:
                                st.session_state.active_sources[sid] = True
                                bar.progress(100)
                                status.update(label="✅ Ingested!", state="complete")
                                st.rerun()
                            else:
                                status.update(label="❌ Failed", state="error")
                                st.error(r.text if r else "Backend unreachable")
                        except Exception as e:
                            status.update(label="❌ Error", state="error")
                            st.error(str(e))

    # ─ Image ───────────────────────────────────────────────────────────────
    with tab_img:
        img_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"], key="img_upload")
        img_sid  = st.text_input("Source ID (optional)", key="img_sid", placeholder="e.g. diagram_ch2")
        if img_file and st.button("⬆️ Ingest Image", use_container_width=True, key="btn_img"):
            sid = img_sid.strip() or img_file.name.replace(" ", "_")[:20]
            with st.status("Captioning image …", expanded=True) as status:
                bar = st.progress(20)
                try:
                    r = _ingest_file(img_file, "image", sid)
                    bar.progress(90)
                    if r and r.ok:
                        st.session_state.active_sources[sid] = True
                        bar.progress(100)
                        status.update(label="✅ Ingested!", state="complete")
                        st.rerun()
                    else:
                        status.update(label="❌ Failed", state="error")
                        st.error(r.text if r else "Backend unreachable")
                except Exception as e:
                    status.update(label="❌ Error", state="error")
                    st.error(str(e))

    st.divider()

    # ── 5. Active Sources (with per-source checkboxes) ──────────────────
    st.subheader("📂 Active Sources")
    sources = _fetch_sources()
    if not sources:
        st.caption("No sources yet — add one above.")
    for s in sources:
        sid  = s.get("source_id", s.get("id", "unknown"))
        name = str(s.get("name", sid))[:30]
        stype = s.get("type", "")
        type_icon = {
            "pdf": "📄", "youtube": "🎥", "website": "🌐",
            "text": "📝", "image": "🖼️",
        }.get(stype, "📦")

        # Default new sources to True (active)
        st.session_state.active_sources.setdefault(sid, True)

        col_cb, col_del = st.columns([5, 1])
        checked = col_cb.checkbox(
            f"{type_icon} {name}",
            value=st.session_state.active_sources[sid],
            key=f"src_cb_{sid}",
        )
        st.session_state.active_sources[sid] = checked

        if col_del.button("🗑", key=f"del_{sid}", help=f"Delete {name}"):
            r = _delete(f"/sources/{sid}")
            if r and r.ok:
                st.session_state.active_sources.pop(sid, None)
                st.rerun()
            else:
                st.error("Delete failed")

    # Show how many sources are active in the query
    active = _active_source_ids()
    if sources:
        st.caption(
            f"🔍 Querying {len(active)}/{len(sources)} source(s)"
            if active else "⚠️ All sources deselected — using all"
        )

    st.divider()

    # ── 6. RAGAS sidebar summary ─────────────────────────────────────
    if _HAS_UI:
        try:
            hist = requests.get(f"{API_URL}/ragas/history?limit=1", timeout=3).json()
            last = (hist.get("history") or [None])[0]
            if last:
                show_ragas_sidebar(last)
        except Exception:
            pass


# ── Main chat area ──────────────────────────────────────────────────────────────

mode_labels = {"chat": "💬 Chat", "deep_research": "🔬 Deep Research", "study": "🎓 Study"}
st.header(mode_labels.get(st.session_state.current_mode, st.session_state.current_mode))

# Show active source chips
active_ids = _active_source_ids()
if active_ids:
    st.caption("Searching: " + " · ".join(f"`{sid}`" for sid in active_ids))
else:
    st.caption("🔍 Searching all ingested sources")

# ── Render chat history ────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        meta  = msg.get("metadata", {})
        ragas = msg.get("ragas")

        if ragas and _HAS_UI:
            render_grounding_badge(ragas)
        elif ragas:
            st.caption(f"📎 Faithfulness {ragas.get('faithfulness', 0)*100:.0f}%  |  Grade: {ragas.get('grade','N/A')}")

        # Citations
        citations = meta.get("citations", [])
        if citations:
            with st.expander(f"📎 {len(citations)} citation(s)"):
                for c in citations:
                    st.caption(f"**[{c.get('label', '?')}]** {c.get('content', '')[:200]}")

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


# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask your documents anything …"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder  = st.empty()
        full_answer  = ""
        metadata: Dict[str, Any] = {}
        ragas_result: Optional[dict] = None

        active_ids = _active_source_ids()   # source_ids for this query

        payload = {
            "query":       prompt,
            "mode":        st.session_state.current_mode,
            "stream":      True,
            "temperature": st.session_state.temperature,
            "top_p":       st.session_state.top_p,
            "source_ids":  active_ids,       # NEW: forwarded to backend
        }

        try:
            with requests.post(
                f"{API_URL}/query/stream", json=payload, stream=True, timeout=120
            ) as r:
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

            if ragas_result and _HAS_UI:
                render_grounding_badge(ragas_result)

            # Citations inline
            citations = metadata.get("citations", [])
            if citations:
                with st.expander(f"📎 {len(citations)} citation(s)", expanded=False):
                    for c in citations:
                        st.caption(f"**[{c.get('label','?')}]** {c.get('content','')[:200]}")

            # Follow-ups as buttons
            follow_ups = metadata.get("follow_ups", [])
            if follow_ups:
                st.markdown("**💭 Explore further:**")
                for fq in follow_ups[:3]:
                    st.button(fq, key=f"fq_{hash(fq)}", disabled=True)

            if metadata:
                with st.expander("📋 Details", expanded=False):
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
