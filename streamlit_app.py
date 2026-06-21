"""
streamlit_app.py
-----------------
The user-facing UI. Calls the core pipeline functions DIRECTLY
(not through the Flask API) to keep the demo simple and fast.

Why not go through Flask here?
The Flask API exists to demonstrate API-building skills (per the JD),
but for the interactive UI it's simpler and more responsive to call
the Python functions in-process. In a real product, the UI (could be
React/Next.js) would call the Flask/FastAPI backend over HTTP instead.
This separation is intentional and worth explaining if asked.
"""

import os
import streamlit as st

from core.document_loader import load_and_chunk
from core.vector_store import VectorStore
from core.llm_client import LLMClient
from core.agent_pipeline import DocumentAgentPipeline
from core.logger import get_logger

logger = get_logger(__name__)

st.set_page_config(page_title="Document Intelligence AI Assistant", page_icon="📄", layout="wide")

UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- Streamlit session state: persists data across reruns/interactions ---
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
    st.session_state.chunks = None
    st.session_state.filename = None


@st.cache_resource
def load_llm_client():
    """
    Cached so the Groq client (and API key check) is only created ONCE
    per app session, not on every Streamlit rerun.
    """
    return LLMClient()


# ----------------------------------------------------------------------
# SIDEBAR: Document upload
# ----------------------------------------------------------------------
st.sidebar.title("📄 Document Intelligence AI Assistant")
st.sidebar.markdown(
    "Upload a PDF or TXT document, then ask questions, get a summary, "
    "or extract key points — all powered by a Retrieve → Reason → "
    "Generate pipeline (RAG)."
)

uploaded_file = st.sidebar.file_uploader("Upload a document", type=["pdf", "txt"])

if uploaded_file is not None:
    # Only reprocess if it's a NEW file (avoid re-embedding on every rerun)
    if st.session_state.filename != uploaded_file.name:
        save_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("Reading and chunking document..."):
            try:
                chunks = load_and_chunk(save_path)
            except Exception as e:
                st.sidebar.error(f"Failed to read document: {e}")
                st.stop()

        with st.spinner("Building embeddings + vector index (first run downloads the model, ~1-2 min)..."):
            try:
                vector_store = VectorStore()
                vector_store.build_from_chunks(chunks)
            except Exception as e:
                st.sidebar.error(f"Failed to build vector store: {e}")
                st.stop()

        try:
            llm_client = load_llm_client()
        except ValueError as e:
            st.sidebar.error(str(e))
            st.stop()

        st.session_state.pipeline = DocumentAgentPipeline(vector_store, llm_client)
        st.session_state.chunks = chunks
        st.session_state.filename = uploaded_file.name

        st.sidebar.success(f"✅ '{uploaded_file.name}' processed into {len(chunks)} chunks.")

# ----------------------------------------------------------------------
# MAIN AREA
# ----------------------------------------------------------------------
st.title("Document Intelligence AI Assistant")

if st.session_state.pipeline is None:
    st.info("👈 Upload a PDF or TXT document from the sidebar to get started.")
    st.markdown("""
    **What this app demonstrates:**
    - 🔍 **RAG (Retrieval-Augmented Generation)** — answers are grounded in YOUR document, not the LLM's general knowledge
    - 🧩 **Chunking + Embeddings** — the document is split and converted into searchable vectors
    - 📊 **FAISS similarity search** — finds the most relevant parts of the document for each question
    - 🤖 **Agent-style pipeline** — every request flows through explicit Input → Retrieve → Reason → Generate steps
    """)
    st.stop()

st.success(f"📄 Active document: **{st.session_state.filename}**  ({len(st.session_state.chunks)} chunks)")

tab1, tab2, tab3 = st.tabs(["💬 Ask Questions", "📝 Summary", "🔑 Key Points"])

# --- TAB 1: Q&A ---
with tab1:
    st.subheader("Ask a question about your document")
    question = st.text_input("Your question:", placeholder="e.g. What is the main topic of this document?")

    if st.button("Ask", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Retrieving relevant context and generating answer..."):
                try:
                    result = st.session_state.pipeline.answer_question(question)
                except Exception as e:
                    st.error(f"Something went wrong: {e}")
                    result = None

            if result:
                st.markdown("### Answer")
                st.write(result["answer"])

                with st.expander("🔍 See the agent's steps + retrieved context (transparency)"):
                    st.write("**Pipeline steps executed:**", " → ".join(result["steps"]))
                    st.write("**Top retrieved chunks used as context:**")
                    for r in result["retrieved_chunks"]:
                        st.markdown(f"**Rank {r['rank']}** (distance: `{r['distance']:.4f}`)")
                        st.text(r["chunk"][:400] + ("..." if len(r["chunk"]) > 400 else ""))
                        st.divider()

# --- TAB 2: Summary ---
with tab2:
    st.subheader("Generate a summary of the entire document")
    if st.button("Generate Summary", type="primary"):
        with st.spinner("Summarizing document (this may take a moment for longer docs)..."):
            try:
                summary = st.session_state.pipeline.summarize_document(st.session_state.chunks)
                st.markdown("### Summary")
                st.write(summary)
            except Exception as e:
                st.error(f"Something went wrong: {e}")

# --- TAB 3: Key Points ---
with tab3:
    st.subheader("Extract key points from the document")
    num_points = st.slider("Number of key points", min_value=3, max_value=10, value=6)
    if st.button("Extract Key Points", type="primary"):
        with st.spinner("Extracting key points..."):
            try:
                points = st.session_state.pipeline.extract_key_points(st.session_state.chunks, num_points)
                st.markdown("### Key Points")
                st.write(points)
            except Exception as e:
                st.error(f"Something went wrong: {e}")
