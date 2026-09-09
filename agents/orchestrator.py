"""Choose metadata lookup or semantic RAG for each user question."""

from __future__ import annotations

from typing import Any

from agents.rag_agent import answer_question


STRUCTURE_KEYWORDS = {"language", "languages", "contributor", "contributors", "commit", "commits", "structure", "file tree", "files", "directory", "directories", "repository size"}


def route_question(question: str, repo_metadata: dict[str, Any], rag_index: Any, rag_chunks: list[dict[str, str]], api_key: str) -> dict[str, Any]:
    """Route simple repository facts to the analyst; all else to the RAG agent."""
    lowered = question.lower()
    if any(keyword in lowered for keyword in STRUCTURE_KEYWORDS):
        parts: list[str] = []
        if "language" in lowered:
            languages = repo_metadata.get("languages", {})
            parts.append("Languages: " + (", ".join(languages) if languages else "not available"))
        if "contributor" in lowered:
            people = repo_metadata.get("contributors", [])
            parts.append("Contributors: " + (", ".join(p["login"] for p in people[:10]) if people else "not available"))
        if "commit" in lowered:
            parts.append(f"Commit count: {repo_metadata.get('commit_count', 'not available')}")
        if any(word in lowered for word in ("structure", "file tree", "files", "directory", "directories", "repository size")):
            stats = repo_metadata.get("file_statistics", {})
            wants_python = "python" in lowered
            wants_tests = "test" in lowered
            wants_top_level_dirs = "top-level" in lowered or "top level" in lowered

            if wants_python:
                parts.append(f"Python files: {stats.get('python_files', 0)}")
            if wants_tests:
                parts.append(f"Test files: {stats.get('test_files', 0)}")
            if wants_top_level_dirs:
                directories = stats.get("top_level_directories", [])
                parts.append(
                    f"Top-level directories ({len(directories)}): "
                    + (", ".join(directories) if directories else "none")
                )
            if not (wants_python or wants_tests or wants_top_level_dirs):
                total = stats.get("total_files", len(repo_metadata.get("file_tree", [])))
                parts.append(f"The repository contains {total} files.")
        return {"agent": "Repo Analyst Agent", "answer": "\n\n".join(parts) or "I can provide repository metadata.", "sources": []}

    result = answer_question(question, rag_index, rag_chunks, api_key)
    return {"agent": "RAG Agent", **result}
