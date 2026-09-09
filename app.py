"""Streamlit interface for RepoMind Lite."""

import os

import streamlit as st
from dotenv import load_dotenv

from agents.orchestrator import route_question
from agents.rag_agent import build_index
from agents.repo_analyst import analyze_repo


load_dotenv()
st.set_page_config(page_title="RepoMind Lite", page_icon="🔎")
st.title("🔎 RepoMind Lite")
st.caption("Ask a Repo Analyst Agent for repository facts or a RAG Agent about its code.")

if "history" not in st.session_state:
    st.session_state.history = []

with st.form("analyze_form"):
    github_url = st.text_input("GitHub repository URL", placeholder="https://github.com/pallets/flask")
    submitted = st.form_submit_button("Analyze Repo")

if submitted:
    with st.spinner("Fetching repository metadata and cloning source..."):
        metadata = analyze_repo(github_url, os.getenv("GITHUB_TOKEN"))
    if not metadata["success"]:
        st.error(f"Could not analyze the repository: {metadata['error']}")
    else:
        try:
            with st.spinner("Building the semantic code index (first run downloads the embedding model)..."):
                index, chunks = build_index(metadata["repo_path"])
            st.session_state.metadata = metadata
            st.session_state.index = index
            st.session_state.chunks = chunks
            st.session_state.history = []
            st.success(f"Ready: indexed {len(chunks)} chunks from {metadata['repo_name']}.")
        except Exception as error:
            st.error(f"Metadata was fetched, but indexing failed: {error}")

if "metadata" in st.session_state:
    metadata = st.session_state.metadata
    st.subheader("Repository metadata")
    col1, col2 = st.columns(2)
    col1.metric("Commits", metadata["commit_count"])
    col2.metric("Indexed chunks", len(st.session_state.chunks))
    st.dataframe([{"Language": key, "Bytes": value} for key, value in metadata["languages"].items()], hide_index=True)
    st.write("**Top contributors:**", ", ".join(item["login"] for item in metadata["contributors"][:10]) or "Not available")

    for item in st.session_state.history:
        with st.chat_message("user"):
            st.write(item["question"])
        with st.chat_message("assistant"):
            st.caption(item["agent"])
            st.write(item["answer"])
            if item["sources"]:
                st.caption("Sources: " + ", ".join(item["sources"]))

    question = st.chat_input("Ask about this repository")
    if question:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = route_question(question, metadata, st.session_state.index, st.session_state.chunks, os.getenv("GEMINI_API_KEY", ""))
            st.caption(result["agent"])
            st.write(result["answer"])
            if result["sources"]:
                st.caption("Sources: " + ", ".join(result["sources"]))
        st.session_state.history.append({"question": question, **result})
else:
    st.info("Enter a public GitHub repository URL to begin.")
