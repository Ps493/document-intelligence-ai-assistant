"""
config.py
---------
Central place for all settings/constants used across the project.

Why this file exists (interview point):
Instead of hardcoding values like chunk size or model names inside
multiple files, we keep them in ONE place. If we ever want to change
the embedding model or chunk size, we change it here only — this is
a basic but real "separation of concerns" practice.
"""

import os
from dotenv import load_dotenv

# Load variables from a local .env file (if present) into the environment.
# This keeps secrets (like API keys) out of the source code.
load_dotenv()

# --- LLM (Groq) settings ---
# Groq gives a free, fast, OpenAI-compatible API. We read the key from
# the environment instead of hardcoding it (security best practice).
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.1-8b-instant"  # free, fast Groq-hosted model

# --- Embedding model settings ---
# Runs LOCALLY on CPU (no API cost). This converts text into vectors
# (lists of numbers) that capture semantic meaning.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# --- Chunking settings ---
# Documents are split into smaller overlapping pieces before embedding.
# See document_loader.py for WHY this is necessary.
CHUNK_SIZE = 500       # max characters per chunk
CHUNK_OVERLAP = 50     # overlap between consecutive chunks

# --- Retrieval settings ---
TOP_K_RESULTS = 3  # how many chunks to retrieve per question

# --- Think9: relevance threshold ---
# L2 distance above this means "the best match still isn't actually relevant" —
# tuned empirically for all-MiniLM-L6-v2 + short chunks. Below this = confident
# enough to answer directly; above this = flag as low-confidence / decline
# rather than let the LLM guess from weak context. See agent_pipeline.py.
RELEVANCE_DISTANCE_THRESHOLD = 1.0

# --- Logging ---
LOG_FILE_PATH = "logs/app.log"
