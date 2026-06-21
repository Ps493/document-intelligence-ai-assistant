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
from core.logger import get_logger

logger = get_logger(__name__)

app = Flask(__name__)

UPLOAD_FOLDER = "uploaded_files"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Simple in-memory "session" for one active document ---
# (See note above about why this is simplified for the demo.)
state = {
    "vector_store": None,
    "chunks": None,
    "pipeline": None,
    "filename": None,
}

llm_client = LLMClient()  # created once and reused across requests


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "Document Intelligence API is running"})


@app.route("/upload", methods=["POST"])
def upload_document():
    """
    Accepts a file upload (multipart/form-data, field name 'file'),
    saves it, loads + chunks it, and builds a fresh vector store.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request. Use form field 'file'."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    save_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(save_path)
    logger.info(f"File uploaded: {file.filename}")

    try:
        chunks = load_and_chunk(save_path)

        vector_store = VectorStore()
        vector_store.build_from_chunks(chunks)

        state["vector_store"] = vector_store
        state["chunks"] = chunks
        state["pipeline"] = DocumentAgentPipeline(vector_store, llm_client)
        state["filename"] = file.filename

        return jsonify({
            "message": "Document processed successfully.",
            "filename": file.filename,
            "num_chunks": len(chunks),
        }), 200

    except Exception as e:
        logger.error(f"Error processing upload: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/ask", methods=["POST"])
def ask_question():
    """
    Body (JSON): { "question": "What is this document about?" }
    Returns the answer plus the retrieved chunks used as context.
    """
    if state["pipeline"] is None:
        return jsonify({"error": "No document uploaded yet. Call /upload first."}), 400

    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "Field 'question' is required."}), 400

    try:
        result = state["pipeline"].answer_question(question)
        return jsonify(result), 200
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/summarize", methods=["GET"])
def summarize():
    if state["pipeline"] is None:
        return jsonify({"error": "No document uploaded yet. Call /upload first."}), 400

    try:
        summary = state["pipeline"].summarize_document(state["chunks"])
        return jsonify({"summary": summary}), 200
    except Exception as e:
        logger.error(f"Error summarizing document: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/keypoints", methods=["GET"])
def keypoints():
    if state["pipeline"] is None:
        return jsonify({"error": "No document uploaded yet. Call /upload first."}), 400

    try:
        points = state["pipeline"].extract_key_points(state["chunks"])
        return jsonify({"key_points": points}), 200
    except Exception as e:
        logger.error(f"Error extracting key points: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    logger.info("Starting Flask API on http://localhost:5000")
    app.run(debug=True, port=5000)
