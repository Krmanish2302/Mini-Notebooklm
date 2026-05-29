import streamlit as st
import requests
import json

# Point to your local FastAPI backend
API_URL = "http://localhost:8000/api"

st.set_page_config(page_title="Mini NotebookLM", layout="wide", page_icon="📓")

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_mode" not in st.session_state:
    st.session_state.current_mode = "chat"
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.7
if "top_p" not in st.session_state:
    st.session_state.top_p = 1.0

# ==========================================
# SIDEBAR: All features mapped as requested
# ==========================================
with st.sidebar:
    st.title("📓 Mini NotebookLM")
    st.markdown("A local, open-source RAG research assistant.")

    # 1. Configuration
    st.subheader("⚙️ Configuration")
    provider = st.selectbox("LLM Provider", ["groq", "openai", "ollama"], index=0)
    model = st.text_input("Model Name", value="llama-3.3-70b-versatile")
    api_key = st.text_input("API Key", type="password")
    
    if st.button("Update Config"):
        res = requests.post(f"{API_URL}/config", json={"provider": provider, "model": model, "api_key": api_key})
        if res.status_code == 200:
            st.success("Config successfully updated!")
        else:
            st.error(f"Error: {res.text}")

    st.divider()

    # 2. Mode Selection
    st.subheader("🧠 Pipeline Mode")
    mode = st.radio(
        "Select Active Mode", 
        ["chat", "deep_research", "study"], 
        format_func=lambda x: x.replace("_", " ").title()
    )
    if mode != st.session_state.current_mode:
        requests.post(f"{API_URL}/mode", json={"mode": mode})
        st.session_state.current_mode = mode
        st.session_state.messages = [] # Clear chat history on mode switch
        st.rerun()

    # 3. Persona / Tuning
    st.subheader("🎭 Tuning")
    with st.expander("Advanced Inference Settings"):
        st.session_state.temperature = st.slider("Temperature", 0.0, 2.0, st.session_state.temperature)
        st.session_state.top_p = st.slider("Top P", 0.0, 1.0, st.session_state.top_p)

    st.divider()

    # 4. Ingestion / Knowledge Base
    st.subheader("📚 Knowledge Base")
    ingest_type = st.selectbox("Source Type", ["pdf", "website", "youtube", "csv", "text"])
    
    file, url = None, None
    if ingest_type in ["pdf", "csv", "text"]:
        file = st.file_uploader(f"Upload {ingest_type.upper()}")
    else:
        url = st.text_input("Source URL")

    if st.button("Ingest Source"):
        with st.spinner("Ingesting and embedding..."):
            try:
                if file:
                    files = {"file": (file.name, file.getvalue(), file.type)}
                    data = {"source_type": ingest_type}
                    res = requests.post(f"{API_URL}/ingest", files=files, data=data)
                elif url:
                    data = {"url": url, "source_type": ingest_type}
                    res = requests.post(f"{API_URL}/ingest", data=data)
                else:
                    st.warning("Please provide a file or URL.")
                    res = None
                
                if res and res.status_code == 200:
                    st.success("Source successfully ingested!")
                elif res:
                    st.error(f"Error: {res.text}")
            except Exception as e:
                st.error(f"Backend unreachable: {e}")

    # 5. List & Manage Sources
    st.subheader("📂 Active Sources")
    try:
        sources_res = requests.get(f"{API_URL}/sources")
        if sources_res.status_code == 200:
            sources = sources_res.json().get("sources", [])
            if not sources:
                st.caption("No sources ingested yet.")
            for s in sources:
                col1, col2 = st.columns([4, 1])
                col1.caption(f"📄 {s.get('name', 'Unknown')[:20]}...")
                if col2.button("🗑️", key=f"del_{s.get('id')}", help="Delete source"):
                    requests.delete(f"{API_URL}/sources/{s.get('id')}")
                    st.rerun()
    except Exception:
        st.caption("Could not fetch active sources.")

# ==========================================
# MAIN AREA: Chat Interface & Dynamic Outputs
# ==========================================
st.header(f"{st.session_state.current_mode.replace('_', ' ').title()} Mode")

# Render historical messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Render metadata if it exists
        if msg.get("metadata"):
            with st.expander("View Details & Metadata"):
                if "quiz_cards" in msg["metadata"] and msg["metadata"]["quiz_cards"]:
                    st.write("**Quiz Cards Generated:**")
                    st.json(msg["metadata"]["quiz_cards"])
                if "ragas" in msg["metadata"]:
                    st.write("**RAGAS Evaluation:**")
                    st.json(msg["metadata"]["ragas"])

# Input handling
if prompt := st.chat_input("Ask your documents anything or start research..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Show assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        metadata = {}
        
        payload = {
            "query": prompt,
            "mode": st.session_state.current_mode,
            "stream": True,
            "temperature": st.session_state.temperature,
            "top_p": st.session_state.top_p
        }
        
        try:
            # Stream the response via SSE (Server-Sent Events) from FastAPI
            with requests.post(f"{API_URL}/query/stream", json=payload, stream=True) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        if decoded_line.startswith("data: "):
                            data_str = decoded_line[6:]
                            try:
                                data_json = json.loads(data_str)
                                msg_type = data_json.get("type")
                                
                                if msg_type == "token":
                                    full_response += data_json.get("content", "")
                                    message_placeholder.markdown(full_response + "▌")
                                elif msg_type == "metadata":
                                    metadata = data_json
                                elif msg_type == "error":
                                    st.error(data_json.get("detail"))
                            except json.JSONDecodeError:
                                pass
                                
            # Finalize response UI
            message_placeholder.markdown(full_response)
            
            # Beautifully render Deep Research / Study mode metadata outputs
            if metadata:
                with st.expander("View Processing Details & Outputs", expanded=True):
                    if metadata.get("sub_queries"):
                        st.subheader("🔍 Sub-Queries Explored")
                        for sq in metadata["sub_queries"]:
                            st.caption(f"- {sq}")
                    
                    if metadata.get("summary_bullets"):
                        st.subheader("📝 Key Takeaways")
                        for bullet in metadata["summary_bullets"]:
                            st.write(f"• {bullet}")

                    if metadata.get("quiz_cards"):
                        st.subheader("🎓 Flashcards")
                        for i, card in enumerate(metadata["quiz_cards"]):
                            st.markdown(f"**Q:** {card.get('question')}")
                            st.markdown(f"**A:** {card.get('answer')} *(Difficulty: {card.get('difficulty', 'N/A')})*")
                            st.divider()

                    if metadata.get("ragas"):
                        st.subheader("📊 RAGAS Quality Metrics")
                        st.json(metadata["ragas"])

        except requests.exceptions.RequestException as e:
            st.error(f"Error communicating with the backend: {e}")
            
        # Save complete message to state
        st.session_state.messages.append({"role": "assistant", "content": full_response, "metadata": metadata})