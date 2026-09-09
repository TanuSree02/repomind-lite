"""Local embedding index and grounded code-question answering."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import faiss
import google.generativeai as genai
import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"
SUPPORTED_SUFFIXES = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".go", ".rs", ".rb",
    ".php", ".c", ".h", ".cpp", ".cs", ".kt", ".swift", ".html", ".css",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".sh", ".sql", ".md", ".rst", ".txt",
}
SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", "dist", "build"}
_embedder: SentenceTransformer | None = None


def _model() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(MODEL_NAME)
    return _embedder


def _chunk_text(text: str, size: int = 500, overlap: int = 50) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    for start in range(0, len(words), size - overlap):
        chunk = words[start : start + size]
        if chunk:
            chunks.append(" ".join(chunk))
        if start + size >= len(words):
            break
    return chunks


def build_index(repo_path: str) -> tuple[faiss.Index, list[dict[str, str]]]:
    """Build a cosine-similarity FAISS index over readable source/document files."""
    root = Path(repo_path)
    if not root.is_dir():
        raise ValueError(f"Repository directory does not exist: {repo_path}")

    chunks: list[dict[str, str]] = []
    for path in root.rglob("*"):
        if not path.is_file() or any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for number, chunk in enumerate(_chunk_text(text), start=1):
            chunks.append({"source": str(path.relative_to(root)), "text": chunk, "chunk": str(number)})

    if not chunks:
        raise ValueError("No supported code or documentation files were found to index.")

    vectors = _model().encode([item["text"] for item in chunks], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index, chunks


def answer_question(question: str, index: faiss.Index, chunks: list[dict[str, str]], api_key: str) -> dict[str, Any]:
    """Retrieve five chunks and ask Gemini for a source-grounded answer."""
    if not question.strip():
        return {"answer": "Please ask a question.", "sources": []}
    query = _model().encode([question], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query)
    _, positions = index.search(query, min(5, len(chunks)))
    retrieved = [chunks[pos] for pos in positions[0] if pos >= 0]
    sources = list(dict.fromkeys(item["source"] for item in retrieved))
    context = "\n\n".join(f"[Source: {item['source']}]\n{item['text']}" for item in retrieved)

    if not api_key or api_key.startswith("your_"):
        return {
            "answer": "A Gemini API key is required for the generated answer. Relevant files: " + ", ".join(sources),
            "sources": sources,
        }

    try:
        genai.configure(api_key=api_key)
        # Confirmed through Gemini ListModels and a live generation request.
        model = genai.GenerativeModel("models/gemini-3.6-flash")
        response = model.generate_content(
            "Answer only from the provided repository context. Be concise and say when context is insufficient.\n\n"
            f"Question: {question}\n\nRepository context:\n{context}",
            generation_config={"temperature": 0.2},
        )
        return {"answer": response.text or "No answer generated.", "sources": sources}
    except Exception as error:
        return {"answer": f"I retrieved relevant code but could not call Gemini: {error}", "sources": sources}
