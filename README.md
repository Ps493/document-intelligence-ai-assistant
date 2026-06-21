# 📄 Document Intelligence AI Assistant

A modular RAG (Retrieval-Augmented Generation) system that lets you upload a
PDF/TXT document and: ask questions about it, generate a summary, and
extract key points — all grounded in the document's actual content via an
explicit **Input → Retrieve → Reason → Generate** pipeline.

Built as an improved, modular take on a single-file RAG demo — split into
clean, single-responsibility modules with logging, error handling, and both
a Streamlit UI and a Flask API.

---

## ✨ Features

1. **Upload** a PDF or TXT document
2. **Ask questions** — answered using RAG (retrieval-augmented generation), grounded in the document
3. **Generate a summary** of the whole document (map-reduce style)
4. **Extract key points** as a clean bullet list
5. **Agent-style multi-step pipeline** — every request explicitly logs its `INPUT → RETRIEVE → REASON → GENERATE` steps, visible in the UI and in `logs/app.log`

---

## 🧱 Tech Stack

| Layer | Choice | Why |
|---|---|---|
| LLM | **Groq API** (Llama 3.1 8B) | Free tier, fast, OpenAI-compatible API shape |
| Embeddings | **sentence-transformers** (`all-MiniLM-L6-v2`) | Runs locally/free, no API cost, genuinely teaches embeddings |
| Vector store | **FAISS** (`IndexFlatL2`, in-memory) | Industry-standard, simple to explain |
| Backend | **Flask API** | Demonstrates Python API-building skills |
| UI | **Streamlit** | Fast to build, good for live demos |

---

## 📁 Project Structure

```
doc-intelligence-assistant/
├── core/
│   ├── config.py            # All settings/constants in one place
│   ├── logger.py            # Logging setup (console + file)
│   ├── document_loader.py   # PDF/TXT loading + chunking
│   ├── vector_store.py      # Embeddings + FAISS index + similarity search
│   ├── llm_client.py        # Groq API wrapper
│   └── agent_pipeline.py    # The Retrieve→Reason→Generate workflow (core logic)
├── sample_data/
│   └── sample_document.txt  # Test document
├── logs/
│   └── app.log               # Generated at runtime
├── streamlit_app.py          # UI (calls core/ functions directly)
├── api.py                    # Flask API (calls core/ functions over HTTP)
├── requirements.txt
├── .env.example
└── README.md
```

**Why this structure?** Each file has ONE job. `document_loader.py` doesn't
know about embeddings; `vector_store.py` doesn't know about the LLM;
`agent_pipeline.py` orchestrates them together. This is the difference
between a "tutorial script" and something you can confidently extend and
debug — a key thing interviewers look for.

---

## 🚀 How to Run Locally

### 1. Get a free Groq API key
Go to [console.groq.com/keys](https://console.groq.com/keys), sign up free (no credit card), and create an API key.

### 2. Clone/download the project, then set up environment
```bash
cd doc-intelligence-assistant
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Add your API key
```bash
cp .env.example .env
# Open .env and paste your key:
# GROQ_API_KEY=gsk_your_real_key_here
```

### 4. Run the Streamlit app (main demo)
```bash
streamlit run streamlit_app.py
```
Open the URL shown (usually `http://localhost:8501`), upload
`sample_data/sample_document.txt`, and try all 3 tabs.

> First run will download the embedding model (~80MB) — takes a minute or two.

### 5. (Optional) Run the Flask API separately
```bash
python api.py
```
Then in another terminal:
```bash
# Upload a document
curl -X POST -F "file=@sample_data/sample_document.txt" http://localhost:5000/upload

# Ask a question
curl -X POST -H "Content-Type: application/json" \
  -d '{"question": "What are the challenges of remote work?"}' \
  http://localhost:5000/ask

# Get a summary
curl http://localhost:5000/summarize

# Get key points
curl http://localhost:5000/keypoints
```

---

## 🎓 Concepts Explained (for interview prep)

### What is RAG, and how is it implemented here?

**RAG = Retrieval-Augmented Generation.** Instead of relying purely on an
LLM's training data (which can be outdated, generic, or hallucinated), RAG
*retrieves* relevant information from a trusted source (your document) and
*injects it into the prompt* before generation. This grounds the LLM's
answer in real, verifiable content.

Implemented here in `agent_pipeline.py` → `answer_question()`:
1. Document is chunked and embedded ahead of time (`vector_store.py`)
2. User's question is embedded the same way
3. FAISS finds the most semantically similar chunks
4. Those chunks are inserted into the prompt as `CONTEXT`
5. The LLM is instructed to answer **only** from that context

### What are embeddings, and why are they used?

An embedding converts text into a vector (list of numbers) such that
texts with similar *meaning* — not just similar words — end up close
together in that vector space. This lets us search by **meaning**
instead of exact keyword matching. E.g. a question about "isolation
when working from home" can match a chunk about "remote work loneliness"
even though they share almost no exact words.

We use `sentence-transformers` (`all-MiniLM-L6-v2`) — it runs locally
on CPU for free, producing 384-dimensional vectors.

### How does similarity search work?

Once every chunk is a vector, we use **FAISS** to find which stored
vectors are closest to the question's vector, using **L2 (Euclidean)
distance** — literally treating each vector as a point in 384-dimensional
space and measuring straight-line distance. Lower distance = more similar
meaning. We use `IndexFlatL2`, FAISS's simplest index: brute-force exact
search, perfect for portfolio-scale data (at huge scale, you'd reach for
an approximate index like `IndexIVFFlat` or HNSW for speed).

### Why is chunking needed?

1. **Context window limits** — LLMs (and embedding models) can't process
   an entire long document at once.
2. **Embedding quality** — embedding a whole multi-topic document into
   ONE vector blurs its meaning. Smaller, focused chunks produce more
   precise vectors.
3. **Relevance & cost** — retrieval should return only the relevant
   section of a document, not the whole thing, keeping the LLM's prompt
   short, fast, and cheap.

We use overlapping chunks (`CHUNK_SIZE=500`, `CHUNK_OVERLAP=50`) so ideas
split across a chunk boundary aren't lost entirely — implemented in
`document_loader.py` → `chunk_text()`.

### How does this simulate an "AI agent workflow"?

A fully autonomous agent (e.g. a LangChain `AgentExecutor` with a ReAct
loop) lets the LLM itself decide which tool to call, in what order, often
looping until it's satisfied. This project deliberately does **not** do
that — instead it implements a **fixed, explicit pipeline** that mirrors
the same underlying pattern:

```
INPUT → RETRIEVE (tool: vector search) → REASON (build grounded prompt)
      → GENERATE (tool: LLM call) → LOG/RETURN
```

This is genuinely how a lot of real "agentic" systems start in production
— deterministic, observable pipelines — before a team introduces a full
autonomous agent framework, often because explicit pipelines are cheaper,
faster, and far easier to debug than letting an LLM freely decide each
step. **Every step is logged** (see `logs/app.log` and the "agent steps"
expander in the Streamlit UI), so you can literally show an interviewer
what happened at each stage of a request.

**Honest interview framing:** *"I implemented an explicit retrieve-reason-
generate pipeline rather than a fully autonomous agent loop — it's more
predictable, debuggable, and cheaper to run, and it's the same core
pattern that frameworks like LangChain abstract into an agent loop. I
haven't built a dynamic agent loop yet, but I understand the
underlying pattern and I'm comfortable extending this toward one."*

---

## 🛡 Production-mindset touches included

- **Logging** — every pipeline step + errors logged to console and `logs/app.log` (`core/logger.py`)
- **Error handling** — unsupported files, empty questions, missing API keys, and LLM failures all fail gracefully with clear messages instead of crashing
- **Separation of concerns** — config, loading, embedding, LLM calls, and orchestration are all isolated modules
- **Caching** — Streamlit's `@st.cache_resource` avoids reloading the embedding model on every UI interaction

## 🔧 Possible extensions (good answers to "what would you improve?")
- Swap `IndexFlatL2` for `IndexIVFFlat`/HNSW for large-scale retrieval
- Chunk by sentence/paragraph boundaries instead of raw characters
- Add a real LangChain `AgentExecutor` with tool-calling for dynamic step selection
- Persist FAISS index to disk so documents don't need re-embedding each session
- Add per-user/session document state instead of a single global state in `api.py`
- Swap Groq for OpenAI/Azure OpenAI by changing only `llm_client.py` (interface is provider-agnostic)
