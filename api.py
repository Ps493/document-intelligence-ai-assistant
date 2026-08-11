"""
api.py
------
A minimal Flask API exposing the document intelligence pipeline as
HTTP endpoints. This demonstrates the "Python-based APIs for AI
applications" requirement from the job description.

Endpoints:
    POST /upload      -> upload a document, builds the vector store
    POST /ask          -> ask a question about the uploaded document
    GET  /summarize     -> get a summary of the uploaded document
    GET  /keypoints      -> get key points from the uploaded document
    GET  /health        -> simple health check

Run with:
    python api.py
Then test with curl or Postman (see README.md for examples).

NOTE ON STATE:
This demo keeps the document's vector store in a single global variable
for simplicity (one document at a time, single user). A production
version would use a session ID or document ID per request and store
each user's vector store separately (e.g. keyed in Redis or a DB).
That tradeoff is worth mentioning if asked in an interview.
"""

import os
from flask import Flask, request, jsonify

from core.document_loader import load_and_chunk
from core.vector_store import VectorStore
from core.llm_client import LLMClient
from core.agent_pipeline import DocumentAgentPipeline
from core.sourcing_agent import SourcingAgentPipeline
from core.logger import get_logger

logger = get_logger(__name__)

app = Flask(__name__)

UPLOAD_FOLDER = "uploaded_files"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Shared institutional memory across ALL uploaded documents/brands ---
# Think9 change: this used to hold ONE document and get overwritten on every
# /upload. Now it's a single cumulative VectorStore that every /upload call
# ADDS to (via add_chunks), which is what makes this a shared "brain" across
# brands instead of a single-doc demo. `chunks` here tracks the LAST uploaded
# doc's chunks only (used for /summarize and /keypoints, which are still
# single-document operations by design).
vector_store = VectorStore()
llm_client = LLMClient()  # created once and reused across requests
memory_pipeline = DocumentAgentPipeline(vector_store, llm_client)
sourcing_pipeline = SourcingAgentPipeline(vector_store, llm_client)

state = {
    "chunks": None,       # chunks of the most recently uploaded doc (for summarize/keypoints)
    "filename": None,
}


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "Document Intelligence API is running"})


@app.route("/upload", methods=["POST"])
def upload_document():
    """
    Accepts a file upload (multipart/form-data, field name 'file'), plus two
    Think9 additions as form fields: 'brand' and 'function' (e.g. function=
    "vendor_notes" | "playbook" | "meeting_notes"). These get attached as
    metadata to every chunk from this file and ADDED to the shared memory —
    this is what lets one running server hold documents from many brands at
    once and scope retrieval by brand/function later.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request. Use form field 'file'."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    brand = request.form.get("brand", "unspecified")
    function = request.form.get("function", "unspecified")

    save_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(save_path)
    logger.info(f"File uploaded: {file.filename} (brand={brand}, function={function})")

    try:
        chunks = load_and_chunk(save_path)
        chunk_metadata = [
            {"brand": brand, "function": function, "source": file.filename}
            for _ in chunks
        ]

        vector_store.add_chunks(chunks, chunk_metadata)

        state["chunks"] = chunks
        state["filename"] = file.filename

        return jsonify({
            "message": "Document processed and added to shared institutional memory.",
            "filename": file.filename,
            "brand": brand,
            "function": function,
            "num_chunks_added": len(chunks),
            "total_chunks_in_memory": vector_store.index.ntotal,
        }), 200

    except Exception as e:
        logger.error(f"Error processing upload: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/ask", methods=["POST"])
def ask_question():
    """
    Body (JSON): { "question": "...", "brand": "BrandA" (optional) }
    `brand` (Think9 addition) scopes retrieval to that brand only; omit it
    to search across the whole shared memory (portfolio-wide synthesis).
    Returns the answer, retrieved chunks, and a confidence flag — low-
    confidence answers are marked needs_human_review instead of guessed at.
    """
    if vector_store.index is None:
        return jsonify({"error": "No documents uploaded yet. Call /upload first."}), 400

    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    brand = data.get("brand")
    metadata_filter = {"brand": brand} if brand else None

    if not question:
        return jsonify({"error": "Field 'question' is required."}), 400

    try:
        result = memory_pipeline.answer_question(question, metadata_filter=metadata_filter)
        return jsonify(result), 200
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/sourcing_ask", methods=["POST"])
def sourcing_ask():
    """
    Think9 addition — the stub second agent, proving the shared retrieval
    core is pluggable. Body (JSON): { "query": "...", "brand": "BrandA" (optional) }
    """
    if vector_store.index is None:
        return jsonify({"error": "No documents uploaded yet. Call /upload first."}), 400

    data = request.get_json(silent=True) or {}
    query = data.get("query", "").strip()
    brand = data.get("brand")
    metadata_filter = {"brand": brand} if brand else None

    if not query:
        return jsonify({"error": "Field 'query' is required."}), 400

    try:
        result = sourcing_pipeline.analyze_sourcing_query(query, metadata_filter=metadata_filter)
        return jsonify(result), 200
    except Exception as e:
        logger.error(f"Error in sourcing analysis: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/summarize", methods=["GET"])
def summarize():
    if not state["chunks"]:
        return jsonify({"error": "No document uploaded yet. Call /upload first."}), 400

    try:
        summary = memory_pipeline.summarize_document(state["chunks"])
        return jsonify({"summary": summary}), 200
    except Exception as e:
        logger.error(f"Error summarizing document: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/keypoints", methods=["GET"])
def keypoints():
    if not state["chunks"]:
        return jsonify({"error": "No document uploaded yet. Call /upload first."}), 400

    try:
        points = memory_pipeline.extract_key_points(state["chunks"])
        return jsonify({"key_points": points}), 200
    except Exception as e:
        logger.error(f"Error extracting key points: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    logger.info("Starting Flask API on http://localhost:5000")
    app.run(debug=True, port=5000)
