# RepoMind Lite

🔗 Live demo: [https://repomind-lite-6nynxhr9mzuv7kwrw9m2nz.streamlit.app](https://repomind-lite-6nynxhr9mzuv7kwrw9m2nz.streamlit.app)

RepoMind Lite is a two-agent RAG application for exploring GitHub repositories. Give it a repository URL, then ask either factual questions about the repository or questions about the codebase.

## Architecture

- **Repo Analyst Agent** clones the repository and uses the GitHub API to collect languages, contributors, commit count, README text, and a recursive file tree.
- **RAG Agent** chunks supported source and documentation files, embeds them with `all-MiniLM-L6-v2`, stores vectors in FAISS, retrieves the five most relevant chunks, and uses Gemini `gemini-3.6-flash` to answer grounded code questions.
- **Orchestrator** uses a small keyword router: structure, language, contributor, and commit questions go directly to the Repo Analyst Agent; other questions go to the RAG Agent.

## Tech stack

Python, Streamlit, GitHub REST API, GitPython, Sentence Transformers, FAISS, Google Gemini API, and python-dotenv.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

Copy/edit `.env` and set `GEMINI_API_KEY` from Google AI Studio (Gemini's free tier is suitable for light use). A `GITHUB_TOKEN` is optional for public repositories but recommended to avoid GitHub's low unauthenticated rate limit and is required for private repositories you can access.

## Run

```bash
streamlit run app.py
```

Try `https://github.com/pallets/flask`, then ask:

- “What languages does this repo use?”
- “Who are the top contributors and how many commits are there?”
- “What does the routing module do?”
- “How does the application dispatch a request?”

The application shows the agent selected for every answer and lists the retrieved source files for RAG answers.
