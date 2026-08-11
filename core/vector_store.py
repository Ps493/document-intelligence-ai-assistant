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
        self.metadata = []         # parallel list of {brand, function, date, source} dicts per chunk
        self.embedding_dim = None

    def build_from_chunks(self, chunks: list[str], metadata: list[dict] | None = None) -> None:
        """
        Embeds a list of text chunks and builds a fresh FAISS index from them.

        `metadata` (Think9 addition): an optional list, same length as `chunks`,
        where metadata[i] describes chunks[i] — e.g.
            {"brand": "BrandA", "function": "vendor_notes", "date": "2026-07-01", "source": "vendor_call.txt"}
        This is what turns a single-document Q&A tool into a multi-brand
        institutional memory store: every chunk knows which brand/function it
        came from, so retrieval can be scoped or left portfolio-wide on purpose.
        If metadata is omitted, every chunk gets an empty dict (backward compatible
        with the original single-document flow).
        """
        if not chunks:
            raise ValueError("Cannot build vector store from an empty list of chunks.")

        if metadata is None:
            metadata = [{} for _ in chunks]
        if len(metadata) != len(chunks):
            raise ValueError("metadata list must be the same length as chunks list.")

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
        self.metadata = metadata

        logger.info(f"FAISS index built successfully with {self.index.ntotal} vectors "
                    f"(dimension={self.embedding_dim})")

    def add_chunks(self, chunks: list[str], metadata: list[dict] | None = None) -> None:
        """
        Think9 addition: append more chunks to an EXISTING index instead of
        rebuilding from scratch. This is what lets the shared institutional
        memory grow as new brands/documents are ingested, instead of each
        upload wiping the previous one.
        """
        if not chunks:
            return
        if metadata is None:
            metadata = [{} for _ in chunks]
        if len(metadata) != len(chunks):
            raise ValueError("metadata list must be the same length as chunks list.")

        embeddings = self.embedding_model.encode(
            chunks, show_progress_bar=False, convert_to_numpy=True
        ).astype("float32")

        if self.index is None:
            self.embedding_dim = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(self.embedding_dim)

        self.index.add(embeddings)
        self.chunks.extend(chunks)
        self.metadata.extend(metadata)
        logger.info(f"Appended {len(chunks)} chunks. Index now has {self.index.ntotal} vectors.")

    def search(self, query: str, top_k: int = TOP_K_RESULTS, metadata_filter: dict | None = None) -> list[dict]:
        """
        Given a question, returns the top_k most semantically similar
        chunks along with their distance scores (lower distance = more similar).

        `metadata_filter` (Think9 addition): optional dict like {"brand": "BrandA"}.
        When set, only chunks whose metadata matches ALL given keys are eligible —
        this is the brand-scoping mechanism described in the architecture doc.
        Implemented as over-fetch-then-filter (simple and exact for a PoC-scale
        index; a production multi-tenant vector DB would push this filter down
        into the index itself for efficiency at scale).
        """
        if self.index is None:
            raise RuntimeError("Vector store is empty. Call build_from_chunks() first.")

        query_vector = self.embedding_model.encode(
            [query], convert_to_numpy=True
        ).astype("float32")

        # Over-fetch when filtering, since some top matches may get filtered out.
        fetch_k = top_k * 5 if metadata_filter else top_k
        fetch_k = min(fetch_k, self.index.ntotal)
        distances, indices = self.index.search(query_vector, fetch_k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:  # FAISS returns -1 if there are fewer results than requested
                continue
            chunk_meta = self.metadata[idx] if idx < len(self.metadata) else {}

            if metadata_filter:
                if not all(chunk_meta.get(k) == v for k, v in metadata_filter.items()):
                    continue

            results.append({
                "rank": len(results) + 1,
                "chunk": self.chunks[idx],
                "distance": float(dist),
                "metadata": chunk_meta,
            })
            if len(results) >= top_k:
                break

        logger.info(f"Retrieved {len(results)} relevant chunks for query: '{query[:50]}...' "
                    f"(filter={metadata_filter})")
        return results
