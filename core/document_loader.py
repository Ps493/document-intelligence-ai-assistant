"""
document_loader.py
-------------------
Responsible for ONE job: turn a raw uploaded file (PDF or TXT) into a
list of clean text chunks ready for embedding.

----------------------------------------------------------------------
WHY CHUNKING IS NEEDED (interview explanation):
----------------------------------------------------------------------
1. LLMs and embedding models have a limited "context window" — they
   can't process an entire 50-page document in one go.
2. Embedding a whole document into ONE vector would blur its meaning —
   a giant document covers many topics, so one vector can't represent
   it well. Smaller chunks = more precise, focused vectors.
3. Smaller chunks let us retrieve ONLY the relevant part of a document
   for a given question, instead of feeding the LLM irrelevant text.
   This keeps answers accurate and keeps API costs/latency low.

We use "overlapping" chunks (CHUNK_OVERLAP) so we don't accidentally
cut a sentence or idea in half at a chunk boundary — the overlap acts
as a safety net so context isn't lost between chunks.
----------------------------------------------------------------------
"""

import os
from pypdf import PdfReader
from core.config import CHUNK_SIZE, CHUNK_OVERLAP
from core.logger import get_logger

logger = get_logger(__name__)


def load_text_from_file(file_path: str) -> str:
    """
    Reads a PDF or TXT file and returns its raw text content.
    Raises a clear error for unsupported file types.
    """
    extension = os.path.splitext(file_path)[1].lower()
    logger.info(f"Loading file: {file_path} (type: {extension})")

    try:
        if extension == ".pdf":
            return _load_pdf(file_path)
        elif extension == ".txt":
            return _load_txt(file_path)
        else:
            raise ValueError(
                f"Unsupported file type '{extension}'. Please upload a .pdf or .txt file."
            )
    except Exception as e:
        logger.error(f"Failed to load file {file_path}: {e}")
        raise


def _load_pdf(file_path: str) -> str:
    """Extracts text from every page of a PDF and joins it together."""
    reader = PdfReader(file_path)
    text_parts = []
    for page_num, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        text_parts.append(page_text)
    full_text = "\n".join(text_parts)

    if not full_text.strip():
        raise ValueError(
            "No extractable text found in this PDF. It may be a scanned image PDF."
        )
    return full_text


def _load_txt(file_path: str) -> str:
    """Reads a plain text file."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Splits a long string of text into overlapping chunks.

    Example with chunk_size=10, overlap=3:
        text = "ABCDEFGHIJKLMNOP"
        chunk 1: "ABCDEFGHIJ"          (chars 0-10)
        chunk 2: "HIJKLMNOPQ"          (starts at 10-3=7 -> overlaps "HIJ")
        ...and so on.

    This is a simple, explainable character-based chunker — good enough
    for a portfolio project and easy to reason about in an interview.
    (Production systems often chunk by sentence/paragraph boundaries or
    token count instead of raw characters — worth mentioning as a
    possible improvement if asked.)
    """
    if not text or not text.strip():
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:  # skip empty chunks
            chunks.append(chunk)
        # Move the window forward, but step back by `overlap` characters
        # so consecutive chunks share some context.
        start += chunk_size - overlap

    logger.info(f"Document split into {len(chunks)} chunks "
                f"(chunk_size={chunk_size}, overlap={overlap})")
    return chunks


def load_and_chunk(file_path: str) -> list[str]:
    """Convenience function: load a file AND chunk it in one call."""
    raw_text = load_text_from_file(file_path)
    return chunk_text(raw_text)
