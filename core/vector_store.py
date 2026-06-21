"""
vector_store.py
----------------
This file implements the "memory" of our RAG system using embeddings
and FAISS (Facebook AI Similarity Search).

----------------------------------------------------------------------
WHAT ARE EMBEDDINGS? (interview explanation)
----------------------------------------------------------------------
An embedding is a way to convert text into a list of numbers (a vector)
such that texts with SIMILAR MEANING end up with vectors that are
mathematically close together in that number-space.

Example (simplified):
    "The cat sat on the mat"   -> [0.12, -0.45, 0.88, ...]
    "A kitten was on the rug"  -> [0.11, -0.40, 0.85, ...]  (close to above!)
    "Stock market crashed"     -> [0.91,  0.02, -0.30, ...] (far away)

We use the `sentence-transformers` library with the model
"all-MiniLM-L6-v2" — it runs locally on CPU, is free, and is small/fast
while still producing good quality embeddings. This avoids needing a
paid OpenAI embeddings API.

----------------------------------------------------------------------
WHAT IS FAISS AND HOW DOES SIMILARITY SEARCH WORK?
----------------------------------------------------------------------
FAISS is a library (by Meta AI) for storing vectors and quickly finding
the ones most similar to a "query vector".

How it works here, step by step:
  1. We embed every document chunk -> get a list of vectors.
  2. We store all those vectors in a FAISS "index" (like a specialized
     database optimized for vector math).
  3. When the user asks a question, we embed the QUESTION the same way.
  4. FAISS compares the question's vector against all stored chunk
     vectors using a distance metric (we use L2 / Euclidean distance)
     and returns the chunks whose vectors are CLOSEST to the question's
     vector — i.e. the chunks that are most semantically relevant.

This is "Retrieval" — the "R" in RAG.
----------------------------------------------------------------------
"""

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from core.config import EMBEDDING_MODEL_NAME, TOP_K_RESULTS
from core.logger import get_logger

logger = get_logger(__name__)


class VectorStore:
    """
    Wraps a FAISS index + the original text chunks together, so we can
    go from "similar vector found" back to "here is the actual text".
    """

    def __init__(self):
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME} (first run may take a moment to download)")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.index = None          # the FAISS index (created once we know vector size)
        self.chunks = []           # original text, so we can map vector -> text
        self.embedding_dim = None

    def build_from_chunks(self, chunks: list[str]) -> None:
        """
        Embeds a list of text chunks and builds a fresh FAISS index from them.
        Call this once per uploaded document.
        """
        if not chunks:
            raise ValueError("Cannot build vector store from an empty list of chunks.")

        logger.info(f"Embedding {len(chunks)} chunks...")
        embeddings = self.embedding_model.encode(
            chunks,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype("float32")  # FAISS requires float32

        self.embedding_dim = embeddings.shape[1]

        # IndexFlatL2 = brute-force exact search using Euclidean (L2) distance.
        # It's the simplest FAISS index type - perfect for a portfolio project
        # (hundreds-to-thousands of chunks). At massive scale, you'd use an
        # approximate index like IndexIVFFlat or HNSW for speed.
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.index.add(embeddings)
        self.chunks = chunks

        logger.info(f"FAISS index built successfully with {self.index.ntotal} vectors "
                    f"(dimension={self.embedding_dim})")

    def search(self, query: str, top_k: int = TOP_K_RESULTS) -> list[dict]:
        """
        Given a question, returns the top_k most semantically similar
        chunks along with their distance scores (lower distance = more similar).
        """
        if self.index is None:
            raise RuntimeError("Vector store is empty. Call build_from_chunks() first.")

        query_vector = self.embedding_model.encode(
            [query], convert_to_numpy=True
        ).astype("float32")

        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            if idx == -1:  # FAISS returns -1 if there are fewer results than top_k
                continue
            results.append({
                "rank": rank + 1,
                "chunk": self.chunks[idx],
                "distance": float(dist),
            })

        logger.info(f"Retrieved {len(results)} relevant chunks for query: '{query[:50]}...'")
        return results
