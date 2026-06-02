"""
streamlit_app.py — Mini NotebookLM · Streamlit UI (v4)
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTHONUTF8", "1")

import json
from typing import Any, Dict, List, Optional
import requests
import streamlit as st

# ── Optional src.ui components ────────────────────────────────────────────────
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

# Suppress Streamlit "Running..." overlay and fading
st.markdown("""
<style>
    /* Prevent the fading overlay effect when Streamlit reruns */
    [data-testid="stAppViewContainer"] {
        opacity: 1 !important;
        filter: none !important;
    }
    .stApp [data-testid="stHeader"] {
        background-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

# ── Session state defaults ────────────────────────────────────────────────────
_DEFAULTS: Dict[str, Any] = {
    "messages":       [],
    "current_mode":   "chat",
    "temperature":    0.7,
    "top_p":          1.0,
    "persona":        "sagan",
    "tone":           "neutral",
    "length":         "medium",
    "active_sources": {},   # {source_id: True/False}
    "pending_query":  "",
}
for k, v in _DEFAULTS.items():
    st.session_state.setdefault(k, v)


# ── Helpers ───────────────────────────────────────────────────────────────────

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
    return [sid for sid, checked in st.session_state.active_sources.items() if checked]


# ── Caching & Health Helpers ──────────────────────────────────────────────────

def refresh_cached_data():
    st.session_state.backend_ok = _backend_ok()
    st.session_state.sources = _fetch_sources() if st.session_state.backend_ok else []

if "backend_ok" not in st.session_state or "sources" not in st.session_state:
    refresh_cached_data()


# ── Ingest methods ────────────────────────────────────────────────────────────
def _ingest_file(file_obj, source_type, source_id, strategy="paragraph_based", embedding_model=None, source_name=None, start_page=1):
    return requests.post(
        f"{API_URL}/ingest",
        files={"file": (file_obj.name, file_obj.getvalue(), file_obj.type)},
        data={
            "source_type": source_type,
            "source_id": source_id,
            "chunking_strategy": strategy,
            "embedding_model": embedding_model,
            "source_name": source_name,
            "start_page": start_page,
        },
        timeout=180,
    )
def _ingest_url(url, source_type, source_id, source_name=None):
    return requests.post(
        f"{API_URL}/ingest",
        data={"url": url, "source_type": source_type, "source_id": source_id, "source_name": source_name},
        timeout=180,
    )

def _ingest_text(text, source_id, source_name=None):
    return requests.post(
        f"{API_URL}/ingest",
        data={"content": text, "source_type": "text", "source_id": source_id, "source_name": source_name},
        timeout=120,
    )


# ── Grounding Circle UI ───────────────────────────────────────────────────────

def draw_grounding_circle(score: float):
    pct = int(score * 100)
    if pct >= 80:
        color = "#10B981"  # Emerald Green
    elif pct >= 60:
        color = "#F59E0B"  # Amber
    else:
        color = "#EF4444"  # Red
    
    svg_html = f"""
    <div style="display: flex; align-items: center; gap: 12px; margin-top: 8px; margin-bottom: 8px;">
        <svg width="36" height="36" viewBox="0 0 36 36">
            <circle cx="18" cy="18" r="15" fill="none" stroke="#E5E7EB" stroke-width="3px" />
            <circle cx="18" cy="18" r="15" fill="none" stroke="{color}" stroke-width="3px"
                    stroke-dasharray="94.2" stroke-dashoffset="{94.2 - (94.2 * pct / 100)}"
                    stroke-linecap="round" transform="rotate(-90 18 18)" />
            <text x="18" y="21" font-family="sans-serif" font-size="10px" font-weight="bold" fill="#374151" text-anchor="middle">{pct}%</text>
        </svg>
        <span style="font-family: sans-serif; font-size: 13px; font-weight: 500; color: #4B5563;">Grounding (Faithfulness) Score</span>
    </div>
    """
    st.html(svg_html)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    col_title, col_badge = st.columns([4, 1])
    col_title.title("📓 Mini NotebookLM")
    if st.session_state.backend_ok:
        col_badge.markdown("<div style='margin-top:28px;font-size:22px'>🟢</div>", unsafe_allow_html=True)
    else:
        col_badge.markdown("<div style='margin-top:28px;font-size:22px'>🔴</div>", unsafe_allow_html=True)
    st.caption("Local · open-source · RAG research assistant")

    # 1. Pipeline Mode
    st.subheader("🧠 Pipeline Mode")
    mode = st.pills(
        "Mode",
        options=["chat", "deep_research", "study"],
        format_func=lambda x: {
            "chat": "💬 Chat", "deep_research": "🔬 Deep Research", "study": "🎓 Study"
        }.get(x, x),
        label_visibility="collapsed",
        default=st.session_state.current_mode,
        key="sidebar_mode_pills"
    )
    if mode != st.session_state.current_mode:
        _post("/mode", json={"mode": mode})
        st.session_state.current_mode = mode
        st.session_state.messages = []
        st.rerun()

    st.divider()

    # 2. Knowledge Base — 6 tabs
    st.subheader("📚 Knowledge Base")
    tab_pdf, tab_yt, tab_web, tab_agent, tab_txt, tab_img = st.tabs([
        "📄 PDF", "🎥 YouTube", "🌐 Website", "🔎 Agent", "📝 Text", "🖼️ Image"
    ])
    with tab_pdf:
        pdf_file = st.file_uploader("Upload PDF", type=["pdf"], key="pdf_upload")
        if pdf_file:
            start_page = st.number_input("Start Page Offset", min_value=1, value=1, step=1, key="pdf_start_page")
            col_an, col_ing = st.columns(2)
            
            if "last_analyzed_pdf" not in st.session_state or st.session_state.last_analyzed_pdf != pdf_file.name or st.session_state.get("last_analyzed_start_page") != start_page:
                st.session_state.pdf_analysis = None
                
            if col_an.button("🔍 Analyze PDF", use_container_width=True, key="btn_pdf_analyze"):
                with st.spinner("Analyzing PDF content..."):
                    try:
                        r = requests.post(
                            f"{API_URL}/analyze",
                            files={"file": (pdf_file.name, pdf_file.getvalue(), pdf_file.type)},
                            data={"source_type": "pdf", "start_page": start_page},
                            timeout=60
                        )
                        if r.ok:
                            st.session_state.pdf_analysis = r.json()
                            st.session_state.last_analyzed_pdf = pdf_file.name
                            st.session_state.last_analyzed_start_page = start_page
                            st.success("Analysis complete!")
                        else:
                            st.error(f"Analysis failed: {r.text}")
                    except Exception as e:
                        st.error(f"Error analyzing PDF: {e}")
            
            analysis = st.session_state.get("pdf_analysis")
            default_strategy = "paragraph_based"
            embedding_models = []
            
            if analysis:
                rec_strat = analysis.get("recommended_strategy", "paragraph_based")
                if rec_strat == "paragraph":
                    rec_strat = "paragraph_based"
                elif rec_strat == "page":
                    rec_strat = "page_based"
                elif rec_strat == "sentence":
                    rec_strat = "sentence_based"
                default_strategy = rec_strat
                embedding_models = analysis.get("embedding_models", [])
                
                # Check if it's the new pipeline analysis with strategies comparison
                if "strategies" in analysis:
                    cols = st.columns(3)
                    cols[0].metric("Total Pages", analysis.get("total_pages", "N/A"))
                    cols[1].metric("Active Pages", analysis.get("active_pages", "N/A"))
                    cols[2].metric("Estimated Words", f"{analysis.get('total_words_estimated', 0):,}")
                    with st.expander("📄 Statistical Metrics Summary", expanded=False):
                        import pandas as pd
                        df = pd.DataFrame(analysis.get("pages_metrics", []))
                        if not df.empty:
                            # 1. Page-level Metrics describe()
                            df_pages = df[[
                                "page_char_count", "page_word_count", 
                                "page_sentence_count", "page_token_count"
                            ]].rename(columns={
                                "page_char_count": "char_count",
                                "page_word_count": "word_count",
                                "page_sentence_count": "sentence_count",
                                "page_token_count": "token_count"
                            })
                            pages_desc = df_pages.describe()
                            
                            st.write("📊 **Page-Level Metrics Summary**")
                            st.dataframe(pages_desc, use_container_width=True)
                            
                            # 2. Paragraph-level Metrics describe()
                            paragraphs_data = []
                            for page in analysis.get("pages_metrics", []):
                                text = page.get("text", "")
                                paras = [p.strip() for p in text.split("\n\n") if p.strip()]
                                for p in paras:
                                    char_count = len(p)
                                    word_count = len(p.split())
                                    try:
                                        import nltk
                                        s_count = len(nltk.sent_tokenize(p))
                                    except Exception:
                                        s_count = len([s for s in p.split(". ") if s.strip()])
                                    token_count = int(char_count / 4)
                                    paragraphs_data.append({
                                        "char_count": char_count,
                                        "word_count": word_count,
                                        "sentence_count": s_count,
                                        "token_count": token_count
                                    })
                                    
                            if paragraphs_data:
                                df_paras = pd.DataFrame(paragraphs_data)
                                paras_desc = df_paras.describe()
                                st.write("📝 **Paragraph-Level Metrics Summary**")
                                st.dataframe(paras_desc, use_container_width=True)

                            # 3. Sentence-level Metrics describe()
                            sentences_data = []
                            for page in analysis.get("pages_metrics", []):
                                text = page.get("text", "")
                                try:
                                    import nltk
                                    sents = nltk.sent_tokenize(text)
                                except Exception:
                                    sents = [s.strip() for s in text.split(". ") if s.strip()]
                                
                                for s in sents:
                                    s_clean = s.strip()
                                    if not s_clean:
                                        continue
                                    char_count = len(s_clean)
                                    word_count = len(s_clean.split())
                                    sentence_count = 1
                                    token_count = int(char_count / 4)
                                    sentences_data.append({
                                        "char_count": char_count,
                                        "word_count": word_count,
                                        "sentence_count": sentence_count,
                                        "token_count": token_count
                                    })
                                    
                            if sentences_data:
                                df_sents = pd.DataFrame(sentences_data)
                                sents_desc = df_sents.describe()
                                st.write("🔤 **Sentence-Level Metrics Summary**")
                                st.dataframe(sents_desc, use_container_width=True)
                    with st.expander("📊 Strategy Chunk Count Estimates", expanded=True):
                        strats = analysis.get("strategies", {})
                        strat_data = []
                        for key, val in strats.items():
                            strat_data.append({
                                "Strategy": val.get("label", key),
                                "Estimated Chunks": f"{val.get('estimated_chunks', 0):,}",
                                "Description": val.get("description", "")
                            })
                        st.dataframe(pd.DataFrame(strat_data), use_container_width=True, hide_index=True)
                else:
                    with st.expander("📊 Content Analysis Results", expanded=True):
                        st.markdown(f"**Estimated Chunks:** `{analysis.get('chunk_count_estimate', 'N/A')}`")
                        token_stats = analysis.get("token_stats", {})
                        st.markdown(f"**Total Tokens:** `{token_stats.get('total_tokens', 'N/A')}` | **Avg Sentence Length:** `{token_stats.get('avg_sentence_length', 'N/A')} words`")
                        st.markdown(f"**Recommended Strategy:** `{analysis.get('recommended_strategy', 'N/A')}`")
                        if analysis.get("previews"):
                            st.markdown("**Chunk Previews:**")
                            for prev in analysis["previews"]:
                                st.caption(f"Chunk {prev['index']}: {prev['text']}")
            
            strat_list = PDF_STRATEGIES
            strat_idx = strat_list.index(default_strategy) if default_strategy in strat_list else 0
            pdf_strategy = st.selectbox("Chunking strategy", strat_list, index=strat_idx, key="pdf_strategy")
            
            if not embedding_models:
                r_models = _get("/embedding-models")
                if r_models and r_models.ok:
                    embedding_models = r_models.json().get("models", [])
            
            model_options = [m["name"] for m in embedding_models]
            model_labels = {m["name"]: f"{m['label']} ({m['speed']} - {m['note']})" for m in embedding_models}
            pdf_embed_model = st.selectbox("Embedding Model", model_options, format_func=lambda x: model_labels.get(x, x), key="pdf_embed_model")
            
            pdf_sid = st.text_input("Source ID/Name (optional)", key="pdf_sid", placeholder="e.g. lecture_notes")
            
            if col_ing.button("⬆️ Ingest PDF", use_container_width=True, key="btn_pdf"):
                sid = pdf_sid.strip() or pdf_file.name.replace(" ", "_")[:20]
                with st.status(f"Ingesting {pdf_file.name} …", expanded=True) as status:
                    st.write("Uploading & Chunking …"); bar = st.progress(20)
                    try:
                        r = _ingest_file(pdf_file, "pdf", sid, pdf_strategy, pdf_embed_model, source_name=pdf_sid.strip() or pdf_file.name, start_page=start_page)
                        bar.progress(90)
                        if r and r.ok:
                            st.session_state.active_sources[sid] = True
                            bar.progress(100)
                            status.update(label="✅ Ingested!", state="complete")
                            refresh_cached_data()
                            st.rerun()
                        else:
                            status.update(label="❌ Failed", state="error")
                            st.error(r.text if r else "Backend unreachable")
                    except Exception as e:
                        status.update(label="❌ Error", state="error")
                        st.error(str(e))
    with tab_yt:
        yt_url = st.text_input("YouTube URL", placeholder="https://youtu.be/...", key="yt_url")
        yt_sid = st.text_input("Source ID/Name (optional)", key="yt_sid", placeholder="e.g. talk_on_rag")
        if st.button("⬆️ Ingest YouTube", use_container_width=True, key="btn_yt"):
            if not yt_url.strip():
                st.warning("Provide a YouTube URL.")
            else:
                sid = yt_sid.strip() or "yt_" + yt_url.split("v=")[-1][:8]
                with st.status("Fetching transcript …", expanded=True) as status:
                    bar = st.progress(30)
                    try:
                        r = _ingest_url(yt_url, "youtube", sid, source_name=yt_sid.strip() or yt_url)
                        bar.progress(90)
                        if r and r.ok:
                            st.session_state.active_sources[sid] = True
                            bar.progress(100)
                            status.update(label="✅ Ingested!", state="complete")
                            refresh_cached_data()
                            st.rerun()
                        else:
                            status.update(label="❌ Failed", state="error")
                            st.error(r.text if r else "Backend unreachable")
                    except Exception as e:
                        status.update(label="❌ Error", state="error")
                        st.error(str(e))

    with tab_web:
        web_url = st.text_input("Website URL", placeholder="https://example.com", key="web_url")
        web_sid = st.text_input("Source ID/Name (optional)", key="web_sid", placeholder="e.g. openai_blog")
        if st.button("⬆️ Ingest Website", use_container_width=True, key="btn_web"):
            if not web_url.strip():
                st.warning("Provide a URL.")
            else:
                sid = web_sid.strip() or "web_" + web_url.split("//")[-1][:16].replace("/", "_")
                with st.status("Crawling page …", expanded=True) as status:
                    bar = st.progress(30)
                    try:
                        r = _ingest_url(web_url, "website", sid, source_name=web_sid.strip() or web_url)
                        bar.progress(90)
                        if r and r.ok:
                            st.session_state.active_sources[sid] = True
                            bar.progress(100)
                            status.update(label="✅ Ingested!", state="complete")
                            refresh_cached_data()
                            st.rerun()
                        else:
                            status.update(label="❌ Failed", state="error")
                            st.error(r.text if r else "Backend unreachable")
                    except Exception as e:
                        status.update(label="❌ Error", state="error")
                        st.error(str(e))

    with tab_agent:
        st.caption("Search the web via DuckDuckGo and ingest pages directly.")
        agent_query = st.text_input("Search query", placeholder="What is Gemini 1.5 Pro?", key="agent_search_query")
        if st.button("🔍 Search the Web", use_container_width=True, key="btn_agent_search"):
            if not agent_query.strip():
                st.warning("Provide a search query.")
            else:
                with st.spinner("Searching..."):
                    try:
                        r = requests.post(f"{API_URL}/agent/search", json={"query": agent_query, "max_results": 5})
                        if r.ok:
                            st.session_state.agent_results = r.json().get("results", [])
                        else:
                            st.error(f"Search failed: {r.text}")
                    except Exception as e:
                        st.error(f"Error searching: {e}")
        
        agent_results = st.session_state.get("agent_results", [])
        if agent_results:
            st.markdown("**Search Results:**")
            for idx, res in enumerate(agent_results):
                with st.container(border=True):
                    st.markdown(f"**[{res.get('title')}]({res.get('url')})**")
                    st.caption(res.get("snippet"))
                    btn_key = f"btn_ingest_res_{idx}"
                    if st.button("📥 Ingest as Source", key=btn_key, use_container_width=True):
                        with st.spinner("Fetching and ingesting page content..."):
                            try:
                                r_ing = requests.post(
                                    f"{API_URL}/agent/fetch-and-ingest",
                                    json={
                                        "url": res.get("url"),
                                        "source_name": res.get("title")
                                    }
                                )
                                if r_ing.ok:
                                    st.success(f"Successfully ingested: {res.get('title')}")
                                    refresh_cached_data()
                                    st.rerun()
                                else:
                                    st.error(f"Ingestion failed: {r_ing.text}")
                            except Exception as e:
                                st.error(f"Error ingesting: {e}")

    with tab_txt:
        txt_option = st.radio("Input", ["Upload file", "Paste text"], horizontal=True, key="txt_option")
        txt_sid = st.text_input("Source ID/Name (optional)", key="txt_sid", placeholder="e.g. notes_ch3")
        if txt_option == "Upload file":
            txt_file = st.file_uploader("Upload .txt / .md / .csv", type=["txt", "md", "csv"], key="txt_upload")
            if txt_file and st.button("⬆️ Ingest File", use_container_width=True, key="btn_txt_file"):
                sid = txt_sid.strip() or txt_file.name.replace(" ", "_")[:20]
                with st.status("Processing …", expanded=True) as status:
                    bar = st.progress(20)
                    try:
                        r = _ingest_file(txt_file, "text", sid, source_name=txt_sid.strip() or txt_file.name)
                        bar.progress(90)
                        if r and r.ok:
                            st.session_state.active_sources[sid] = True
                            bar.progress(100)
                            status.update(label="✅ Ingested!", state="complete")
                            refresh_cached_data()
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
                            r = _ingest_text(pasted, sid, source_name=txt_sid.strip() or f"Pasted Text ({len(pasted)} chars)")
                            bar.progress(90)
                            if r and r.ok:
                                st.session_state.active_sources[sid] = True
                                bar.progress(100)
                                status.update(label="✅ Ingested!", state="complete")
                                refresh_cached_data()
                                st.rerun()
                            else:
                                status.update(label="❌ Failed", state="error")
                                st.error(r.text if r else "Backend unreachable")
                        except Exception as e:
                            status.update(label="❌ Error", state="error")
                            st.error(str(e))

    with tab_img:
        img_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"], key="img_upload")
        img_sid  = st.text_input("Source ID/Name (optional)", key="img_sid", placeholder="e.g. diagram_ch2")
        if img_file and st.button("⬆️ Ingest Image", use_container_width=True, key="btn_img"):
            sid = img_sid.strip() or img_file.name.replace(" ", "_")[:20]
            with st.status("Captioning image …", expanded=True) as status:
                bar = st.progress(20)
                try:
                    r = _ingest_file(img_file, "image", sid, source_name=img_sid.strip() or img_file.name)
                    bar.progress(90)
                    if r and r.ok:
                        st.session_state.active_sources[sid] = True
                        bar.progress(100)
                        status.update(label="✅ Ingested!", state="complete")
                        refresh_cached_data()
                        st.rerun()
                    else:
                        status.update(label="❌ Failed", state="error")
                        st.error(r.text if r else "Backend unreachable")
                except Exception as e:
                    status.update(label="❌ Error", state="error")
                    st.error(str(e))

    st.divider()

    # 3. Active Sources (checkboxes with delete)
    st.subheader("📂 Active Sources")
    sources = st.session_state.sources
    if not sources:
        st.caption("No sources yet — add one above.")
    for s in sources:
        sid  = s.get("source_id", s.get("id", "unknown"))
        name = str(s.get("name", sid))[:30]
        stype = s.get("type", "")
        type_icon = {"pdf": "📄", "youtube": "🎥", "website": "🌐", "text": "📝", "image": "🖼️"}.get(stype, "📦")
        st.session_state.active_sources.setdefault(sid, True)

        col_cb, col_del = st.columns([5, 1])
        checked = col_cb.checkbox(f"{type_icon} {name}", value=st.session_state.active_sources[sid], key=f"src_cb_{sid}")
        st.session_state.active_sources[sid] = checked
        
        meta = s.get("metadata", {})
        num_chunks = meta.get("num_chunks", 0) if isinstance(meta, dict) else 0
        col_cb.markdown(
            f"<div style='margin-left: 28px; margin-top: -8px; margin-bottom: 8px; font-size: 0.8rem; color: gray;'>"
            f"{num_chunks} chunks &middot; {num_chunks} vectors"
            f"</div>",
            unsafe_allow_html=True
        )

        if col_del.button("🗑", key=f"del_{sid}", help=f"Delete {name}"):
            r = _delete(f"/sources/{sid}")
            if r and r.ok:
                st.session_state.active_sources.pop(sid, None)
                refresh_cached_data()
                st.rerun()
            else:
                st.error("Delete failed")

    active = _active_source_ids()
    if sources:
        st.caption(
            f"🔍 Querying {len(active)}/{len(sources)} source(s)" if active
            else "⚠️ All deselected — using all"
        )

    st.divider()

    # 4. Persona & Style
    st.subheader("🎭 Persona & Style")
    persona = st.selectbox("Persona", ["sagan", "professor", "eli5", "analyst", "socratic", "journalist"])
    tone    = st.selectbox("Tone",    ["neutral", "casual", "formal", "enthusiastic", "stoic"])
    length  = st.selectbox("Length",  ["medium", "short", "long", "bullets"])
    if st.button("Apply Persona", use_container_width=True):
        r = _post("/persona", json={"persona": persona, "tone": tone, "length": length})
        if r and r.ok:
            st.session_state.update(persona=persona, tone=tone, length=length)
            st.success("Persona updated ✓")
        elif r:
            st.error(r.text)

    with st.expander("🔧 Inference Tuning"):
        st.session_state.temperature = st.slider("Temperature", 0.0, 2.0, st.session_state.temperature, 0.05)
        st.session_state.top_p       = st.slider("Top P",       0.0, 1.0, st.session_state.top_p,       0.05)

    st.divider()

    # 5. LLM Configuration
    st.subheader("⚙️ LLM Configuration")
    provider = st.selectbox("Provider", ["groq", "openai", "ollama", "gemini", "anthropic"])
    model    = st.text_input("Model", value="llama-3.3-70b-versatile")
    api_key  = st.text_input("API Key", type="password")
    if st.button("💾 Apply Config", use_container_width=True):
        r = _post("/config", json={"provider": provider, "model": model, "api_key": api_key})
        if r and r.ok:
            st.success("Config updated ✓")
            refresh_cached_data()
        else:
            st.error(r.text if r else "Backend unreachable")

    st.divider()

    # 6. RAGAS Evaluation
    if _HAS_UI:
        try:
            hist = requests.get(f"{API_URL}/ragas/history?limit=1", timeout=3).json()
            last = (hist.get("history") or [None])[0]
            if last:
                show_ragas_sidebar(last)
        except Exception:
            pass


# ── Main area ─────────────────────────────────────────────────────────────────

mode_labels = {"chat": "💬 Chat", "deep_research": "🔬 Deep Research", "study": "🎓 Study"}
st.header(mode_labels.get(st.session_state.current_mode, st.session_state.current_mode))

active_ids = _active_source_ids()
if active_ids:
    st.caption("Searching: " + " · ".join(f"`{sid}`" for sid in active_ids))
else:
    st.caption("🔍 Searching all ingested sources")


# ── Wrap Chat area in st.fragment to isolate reruns ──────────────────────────

@st.fragment
def render_chat_interface():
    # ── Render chat history ───────────────────────────────────────────────────────
    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                st.container()  # Fix ghost message bug.
            st.markdown(msg["content"])
            meta  = msg.get("metadata", {})
            ragas = msg.get("ragas")

            if msg["role"] == "assistant" and ragas:
                if _HAS_UI:
                    render_grounding_badge(ragas)
                # Custom SVG circle gauge
                faithfulness = ragas.get("faithfulness")
                if faithfulness is not None:
                    draw_grounding_circle(faithfulness)

            try:
                citations = meta.get("citations", [])
                if citations:
                    with st.expander(f"📎 {len(citations)} citation(s)"):
                        for c in citations:
                            st.caption(f"**[{c.get('label','?')}]** {c.get('content','')[:200]}")
            except Exception:
                pass

            try:
                if meta:
                    with st.expander("📋 Details"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"**Pipeline Mode:** `{meta.get('pipeline_mode', st.session_state.current_mode).upper()}`")
                            st.markdown(f"**LLM Model:** `{meta.get('model_name', 'N/A')}`")
                            st.markdown(f"**Chunk Strategy:** `{meta.get('chunk_strategy', 'N/A')}`")
                        with col2:
                            ttft = meta.get('ttft_ms', 0)
                            tot = meta.get('total_time_ms', 0)
                            st.markdown(f"**Time to First Token (TTFT):** `{ttft} ms`" if ttft else "**Time to First Token (TTFT):** `N/A`")
                            st.markdown(f"**Total Generation Time:** `{tot / 1000:.2f} s`" if tot else "**Total Generation Time:** `N/A`")
                        
                        st.divider()
                        
                        stats = meta.get("retrieval_stats")
                        if stats:
                            st.markdown("**🔍 Retrieval Pipeline Trace**")
                            if stats.get("cached_hit"):
                                st.info("⚡ **Semantic Cache Hit (90%+ Similarity)**: Returned cached response directly.")
                            else:
                                st.caption(f"• **Dense Vector Candidates (FAISS):** {stats.get('dense_count', 0)}")
                                st.caption(f"• **Sparse Candidate Counts (BM25):** {stats.get('sparse_count', 0)}")
                                st.caption(f"• **History Turn Candidates:** {stats.get('history_count', 0)}")
                                st.caption(f"• **Reciprocal Rank Fusion (RRF) Candidates:** {stats.get('rrf_fused', 0)}")
                                st.caption(f"• **Reranked Candidates (FlashRank):** {stats.get('reranked', 0)}")
                                if "reordered" in stats:
                                    st.caption(f"• **Reordered Chunks (Lost-in-the-Middle):** {stats.get('reordered', 0)}")
                                else:
                                    st.caption(f"• **Compressed Sentences (Contextual Compressor):** {stats.get('compressed', 0)}")
                            st.divider()

                        if meta.get("pipeline_mode") in ("deep_research", "study"):
                            st.markdown("**📂 Parent-Child Context Resolution**")
                            st.caption("• **SQLite Parent Store:** Fine-grained retrieved child chunks were resolved to whole parent sections/pages in SQLite to preserve full context.")
                            
                        if meta.get("graph_context"):
                            st.markdown("**🌐 SQLite Knowledge Graph Concept Map**")
                            for rel in meta["graph_context"]:
                                st.caption(f"• `{rel['source']}` \u2192 `{rel['target']}` ({rel['relation']}) [Confidence: {rel.get('confidence', 1.0)}]")
                            st.divider()

                        if meta.get("sub_queries"):
                            st.markdown("**🔍 Sub-queries**")
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
                            render_ragas_panel(ragas, key=f"hist_{idx}")

            except Exception:
                pass

    # ── Chat input + follow-up injection ─────────────────────────────────────────
    _pending = st.session_state.get("pending_query", "")
    prompt = st.chat_input("Ask your documents anything …", key="main_chat_input") or _pending
    if _pending:
        st.session_state.pending_query = ""

    if prompt:
        # Escape LaTeX $ signs
        prompt_escaped = prompt.replace("$", r"\$")
        st.session_state.messages.append({"role": "user", "content": prompt_escaped})
        with st.chat_message("user"):
            st.markdown(prompt_escaped)

        with st.chat_message("assistant"):
            st.container()  # Fix ghost message bug.
            placeholder  = st.empty()
            full_answer  = ""
            metadata: Dict[str, Any] = {}
            ragas_result: Optional[dict] = None
            stream_error: Optional[str] = None

            active_ids = _active_source_ids()

            payload = {
                "query":       prompt,
                "mode":        st.session_state.current_mode,
                "stream":      True,
                "temperature": st.session_state.temperature,
                "top_p":       st.session_state.top_p,
                "source_ids":  active_ids,
            }

            # Streaming block — catch ALL exceptions to never block rendering
            try:
                spinner_html = """
                <div class="loader-container" style="display: flex; align-items: center; gap: 12px; padding: 10px 16px; border-radius: 12px; background: rgba(128, 128, 128, 0.05); border: 1px solid rgba(128, 128, 128, 0.1); width: fit-content; margin-bottom: 12px;">
                    <div class="spinner" style="width: 20px; height: 20px; border: 2px solid transparent; border-top: 2px solid #6366F1; border-right: 2px solid #8B5CF6; border-radius: 50%; animation: spin 0.8s cubic-bezier(0.4, 0, 0.2, 1) infinite;"></div>
                    <span class="loading-text" style="font-family: 'Outfit', 'Inter', sans-serif; font-size: 14px; color: var(--text-color, #4B5563); font-weight: 500; animation: pulse 1.5s infinite ease-in-out;">Retrieving knowledge & thinking...</span>
                </div>
                <style>
                    @keyframes spin {
                        0% { transform: rotate(0deg); }
                        100% { transform: rotate(360deg); }
                    }
                    @keyframes pulse {
                        0%, 100% { opacity: 0.6; }
                        50% { opacity: 1; }
                    }
                </style>
                """
                placeholder.markdown(spinner_html, unsafe_allow_html=True)
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
                            stream_error = event.get("detail", "Unknown error")
                        elif etype == "done":
                            break

            except Exception as e:
                stream_error = f"Backend error: {e}"

            # Render final answer
            placeholder.markdown(full_answer if full_answer else "_No response received._")

            if stream_error:
                st.error(stream_error)

            # Grounding circle and inline badge
            if ragas_result:
                if _HAS_UI:
                    render_grounding_badge(ragas_result)
                faithfulness = ragas_result.get("faithfulness")
                if faithfulness is not None:
                    draw_grounding_circle(faithfulness)

            # Citations
            try:
                citations = metadata.get("citations", [])
                if citations:
                    with st.expander(f"📎 {len(citations)} citation(s)", expanded=False):
                        for c in citations:
                            st.caption(f"**[{c.get('label','?')}]** {c.get('content','')[:200]}")
            except Exception:
                pass

            # Follow-up buttons
            try:
                follow_ups = metadata.get("follow_ups", [])
                if follow_ups:
                    st.markdown("**💭 Explore further:**")
                    cols = st.columns(min(len(follow_ups), 3))
                    for i, fq in enumerate(follow_ups[:3]):
                        if cols[i].button(fq, key=f"fq_{len(st.session_state.messages)}_{i}"):
                            st.session_state.pending_query = fq
                            st.rerun()
            except Exception:
                pass

            # Details expander
            try:
                if metadata:
                    with st.expander("📋 Details", expanded=False):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"**Pipeline Mode:** `{metadata.get('pipeline_mode', st.session_state.current_mode).upper()}`")
                            st.markdown(f"**LLM Model:** `{metadata.get('model_name', 'N/A')}`")
                            st.markdown(f"**Chunk Strategy:** `{metadata.get('chunk_strategy', 'N/A')}`")
                        with col2:
                            ttft = metadata.get('ttft_ms', 0)
                            tot = metadata.get('total_time_ms', 0)
                            st.markdown(f"**Time to First Token (TTFT):** `{ttft} ms`" if ttft else "**Time to First Token (TTFT):** `N/A`")
                            st.markdown(f"**Total Generation Time:** `{tot / 1000:.2f} s`" if tot else "**Total Generation Time:** `N/A`")
                        
                        st.divider()

                        stats = metadata.get("retrieval_stats")
                        if stats:
                            st.markdown("**🔍 Retrieval Pipeline Trace**")
                            if stats.get("cached_hit"):
                                st.info("⚡ **Semantic Cache Hit (90%+ Similarity)**: Returned cached response directly.")
                            else:
                                st.caption(f"• **Dense Vector Candidates (FAISS):** {stats.get('dense_count', 0)}")
                                st.caption(f"• **Sparse Candidate Counts (BM25):** {stats.get('sparse_count', 0)}")
                                st.caption(f"• **History Turn Candidates:** {stats.get('history_count', 0)}")
                                st.caption(f"• **Reciprocal Rank Fusion (RRF) Candidates:** {stats.get('rrf_fused', 0)}")
                                st.caption(f"• **Reranked Candidates (FlashRank):** {stats.get('reranked', 0)}")
                                if "reordered" in stats:
                                    st.caption(f"• **Reordered Chunks (Lost-in-the-Middle):** {stats.get('reordered', 0)}")
                                else:
                                    st.caption(f"• **Compressed Sentences (Contextual Compressor):** {stats.get('compressed', 0)}")
                            st.divider()

                        if metadata.get("sub_queries"):
                            st.markdown("**🔍 Sub-queries**")
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
            except Exception:
                pass

            st.session_state.messages.append({
                "role":     "assistant",
                "content":  full_answer,
                "metadata": metadata,
                "ragas":    ragas_result,
            })

# Render the interface
render_chat_interface()
