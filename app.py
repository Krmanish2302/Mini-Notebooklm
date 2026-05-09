import streamlit as st
from src.master_pipeline import MasterPipeline
import os

st.set_page_config(page_title="Mini NotebookLM", layout="wide")

# Initialize session state
if "pipeline" not in st.session_state:
    st.session_state.pipeline = MasterPipeline(mode="chat")
    st.session_state.messages = []
    st.session_state.sources = []

# Top bar
col1, col2, col3 = st.columns([2, 3, 2])

with col1:
    st.title("Mini NotebookLM")

with col2:
    mode = st.radio(
        "Mode",
        options=["Chat", "Deep Research", "Study Mode"],
        horizontal=True
    )
    mode_map = {"Chat": "chat", "Deep Research": "deep_research", "Study Mode": "study"}
    st.session_state.pipeline.set_mode(mode_map[mode])

with col3:
    provider = st.selectbox("Provider", ["Groq", "Ollama", "OpenAI", "Gemini"])
    api_key = st.text_input("API Key", type="password")
    
    if api_key:
        st.session_state.pipeline.set_llm(
            provider=provider.lower(),
            model="llama-3.1-70b-versatile" if provider == "Groq" else "gpt-4",
            api_key=api_key
        )

# Main layout
left_col, main_col = st.columns([1, 3])

# Left sidebar - Sources
with left_col:
    st.header("📚 Sources")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload File",
        type=["pdf", "txt", "csv", "png", "jpg", "mp4"]
    )
    if uploaded_file:
        with st.spinner("Processing..."):
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            source_id = st.session_state.pipeline.ingest(file_path=temp_path)
            st.success(f"Added: {uploaded_file.name}")
            os.remove(temp_path)
    
    # Web search
    st.subheader("🔍 Web Search")
    web_query = st.text_input("Search query")
    if st.button("Search", use_container_width=True) and web_query:
        with st.spinner("Searching..."):
            results = st.session_state.pipeline.web_search.search_and_format(web_query)
            st.session_state.web_results = results
    
    # Display web results with checkboxes
    if "web_results" in st.session_state:
        st.write("Select sources to add:")
        for result in st.session_state.web_results:
            if st.checkbox(result["title"], key=result["id"]):
                # Add selected source
                pass
    
    # Source list
    st.subheader("Added Sources")
    sources = st.session_state.pipeline.source_manager.get_all_sources()
    for source in sources:
        col_a, col_b = st.columns([4, 1])
        with col_a:
            st.write(f"• {source.get('title', 'Untitled')[:30]}...")
        with col_b:
            if st.button("🗑️", key=f"del_{source['id']}"):
                st.session_state.pipeline.source_manager.remove_source(source["id"])
                st.rerun()

# Main chat area
with main_col:
    st.header(f"💬 {mode} Mode")
    
    # Display messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask anything..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            if not st.session_state.pipeline.llm:
                st.warning("Please enter API Key")
            else:
                with st.spinner("Thinking..."):
                    response = st.session_state.pipeline.generate(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

# Bottom settings
with st.expander("⚙️ Settings", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        temp = st.slider("Temperature", 0.0, 2.0, 0.7)
    with col2:
        top_p = st.slider("Top-p", 0.0, 1.0, 0.9)
    with col3:
        max_tokens = st.slider("Max Tokens", 128, 4096, 1024)
    
    if st.button("Apply"):
        if st.session_state.pipeline.llm:
            st.session_state.pipeline.llm.update_tuning(
                temperature=temp, top_p=top_p, max_tokens=max_tokens
            )
            st.success("Settings applied!")
    
    if st.button("🕸️ Show Knowledge Graph"):
        # Display graph visualization
        pass