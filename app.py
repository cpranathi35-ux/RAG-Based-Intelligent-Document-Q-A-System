"""
app.py
------
Streamlit frontend for the Finance RAG Q&A System.
Run with: streamlit run app.py
"""

import os
import tempfile
import streamlit as st
from pathlib import Path
from rag_pipeline import FinanceRAGPipeline

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="FinanceRAG — Stock Report Q&A",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
    }
    .subtitle {
        font-size: 1rem;
        color: #555;
        margin-bottom: 2rem;
    }
    .answer-box {
        background: #f0f7ff;
        border-left: 4px solid #2563eb;
        padding: 1.2rem;
        border-radius: 6px;
        margin: 1rem 0;
    }
    .source-badge {
        background: #e0f2fe;
        color: #0369a1;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        margin-right: 6px;
    }
    .chunk-box {
        background: #fafafa;
        border: 1px solid #e5e7eb;
        padding: 0.8rem;
        border-radius: 6px;
        font-size: 0.85rem;
        color: #444;
        margin-bottom: 0.5rem;
    }
    .stat-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "docs_loaded" not in st.session_state:
    st.session_state.docs_loaded = []
if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0


# ──────────────────────────────────────────────
# SIDEBAR — CONFIGURATION
# ──────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=60)
    st.markdown("## FinanceRAG")
    st.markdown("*AI-powered financial document analysis*")
    st.divider()

    # API Key
    api_key = st.text_input(
        "🔑 OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="Get your key at platform.openai.com",
    )

    model_choice = st.selectbox(
        "🤖 Model",
        ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
        index=0,
        help="GPT-4 gives better analysis but costs more.",
    )

    st.divider()

    # Initialize pipeline
    if api_key and st.button("✅ Initialize Pipeline", use_container_width=True):
        with st.spinner("Setting up RAG pipeline..."):
            st.session_state.pipeline = FinanceRAGPipeline(
                api_key=api_key,
                model=model_choice,
                temperature=0.0,
            )
        st.success("Pipeline ready!")

    st.divider()

    # Document Upload
    st.markdown("### 📂 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload stock reports, 10-Ks, earnings PDFs",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.session_state.pipeline:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.docs_loaded:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=Path(uploaded_file.name).suffix
                    ) as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name
                    chunks = st.session_state.pipeline.ingest_file(tmp_path)
                    st.session_state.docs_loaded.append(uploaded_file.name)
                    st.session_state.total_chunks += chunks
                    os.unlink(tmp_path)
                st.success(f"✓ {uploaded_file.name} — {chunks} chunks")

    # Paste Text Option
    st.divider()
    st.markdown("### 📋 Or Paste Text")
    pasted_text = st.text_area(
        "Paste earnings text, analyst notes, etc.",
        height=120,
        placeholder="Paste financial text here..."
    )
    paste_source = st.text_input("Source label", value="pasted_report")

    if pasted_text and st.button("📥 Load Text", use_container_width=True):
        if st.session_state.pipeline:
            chunks = st.session_state.pipeline.ingest_text(pasted_text, paste_source)
            st.session_state.total_chunks += chunks
            st.success(f"Loaded {chunks} chunks from pasted text.")
        else:
            st.warning("Initialize the pipeline first.")

    # Stats
    if st.session_state.pipeline:
        st.divider()
        st.markdown("### 📊 Index Stats")
        col1, col2 = st.columns(2)
        col1.metric("Documents", len(st.session_state.docs_loaded))
        col2.metric("Chunks", st.session_state.total_chunks)

    # Reset
    st.divider()
    if st.button("🗑️ Reset Everything", use_container_width=True):
        if st.session_state.pipeline:
            st.session_state.pipeline.reset()
        st.session_state.chat_history = []
        st.session_state.docs_loaded = []
        st.session_state.total_chunks = 0
        st.rerun()


# ──────────────────────────────────────────────
# MAIN PANEL
# ──────────────────────────────────────────────
st.markdown('<div class="main-title">📈 FinanceRAG — Stock Report Q&A</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Ask questions about your uploaded financial documents using AI-powered retrieval.</div>', unsafe_allow_html=True)

# Suggested questions
st.markdown("#### 💡 Try asking:")
suggestions = [
    "What was the total revenue for the last fiscal year?",
    "What are the key risk factors mentioned?",
    "How did EPS change compared to the previous quarter?",
    "What is the company's guidance for next quarter?",
    "Summarize the cash flow statement.",
]
cols = st.columns(len(suggestions))
for i, suggestion in enumerate(suggestions):
    if cols[i].button(suggestion, key=f"sug_{i}", use_container_width=True):
        st.session_state["prefilled_query"] = suggestion

st.divider()

# Query Input
prefilled = st.session_state.pop("prefilled_query", "")
question = st.text_input(
    "🔍 Ask a question about your financial documents",
    value=prefilled,
    placeholder="e.g. What was the net income in Q3 2023?",
)

col_ask, col_clear = st.columns([1, 5])
ask_btn = col_ask.button("Ask →", type="primary", use_container_width=True)
if col_clear.button("Clear History", use_container_width=False):
    st.session_state.chat_history = []
    st.rerun()

# Run Query
if ask_btn and question:
    if not st.session_state.pipeline:
        st.error("⚠️ Please enter your API key and initialize the pipeline in the sidebar.")
    elif st.session_state.total_chunks == 0:
        st.warning("📂 Please upload at least one financial document first.")
    else:
        with st.spinner("🔍 Searching documents and generating answer..."):
            try:
                result = st.session_state.pipeline.query(question)
                st.session_state.chat_history.append({
                    "question": question,
                    "answer": result["answer"],
                    "sources": result["sources"],
                    "chunks": result["chunks"],
                })
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Display Chat History
if st.session_state.chat_history:
    for i, entry in enumerate(reversed(st.session_state.chat_history)):
        st.markdown(f"**Q: {entry['question']}**")

        st.markdown(f"""
        <div class="answer-box">
            <strong>Answer:</strong><br>{entry['answer']}
        </div>
        """, unsafe_allow_html=True)

        # Sources
        if entry["sources"]:
            st.markdown("**Sources:**")
            src_html = " ".join([
                f'<span class="source-badge">📄 {s.split("/")[-1]}</span>'
                for s in entry["sources"]
            ])
            st.markdown(src_html, unsafe_allow_html=True)

        # Retrieved Chunks Expander
        with st.expander("🔎 View retrieved context chunks"):
            for j, chunk in enumerate(entry["chunks"]):
                st.markdown(f"""
                <div class="chunk-box">
                    <strong>Chunk {j+1}:</strong><br>{chunk}...
                </div>
                """, unsafe_allow_html=True)

        if i < len(st.session_state.chat_history) - 1:
            st.divider()

else:
    # Empty state
    st.markdown("""
    <div style="text-align:center; padding: 4rem; color: #999;">
        <div style="font-size: 3rem;">📄</div>
        <p>Upload a financial document and ask your first question to get started.</p>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center; color:#aaa; font-size:0.8rem;'>"
    "FinanceRAG v1.0 | Built with LangChain + OpenAI + FAISS | NJIT Data Science Project 2023–2024"
    "</div>",
    unsafe_allow_html=True
)
