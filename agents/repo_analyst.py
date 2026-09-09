"""GitHub metadata collection and local repository cloning."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests
from git import Repo


GITHUB_API = "https://api.github.com"


def _parse_github_url(url: str) -> tuple[str, str]:
    """Extract ``owner, repository`` from common GitHub URL forms."""
    match = re.search(r"github\.com[/:]([^/\s]+)/([^/\s#]+)", url.strip())
    if not match:
        raise ValueError("Please enter a valid GitHub repository URL.")
    owner, repo = match.groups()
    return owner, repo.removesuffix(".git")


def _headers(token: str | None) -> dict[str, str]:
    headers = {"Accept": "application/vnd.github+json"}
    if token and not token.startswith("your_"):
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _request(url: str, headers: dict[str, str]) -> requests.Response:
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    return response


def _all_contributors(api_base: str, headers: dict[str, str]) -> list[dict[str, Any]]:
    contributors: list[dict[str, Any]] = []
    page = 1
    while True:
        result = _request(f"{api_base}/contributors?per_page=100&page={page}", headers).json()
        if not result:
            return contributors
        contributors.extend(
            {"login": item.get("login", "unknown"), "contributions": item.get("contributions", 0)}
            for item in result
        )
        if len(result) < 100:
            return contributors
        page += 1


def _commit_count(api_base: str, headers: dict[str, str]) -> int:
    """Count commits from GitHub's paginated commits endpoint."""
    first = _request(f"{api_base}/commits?per_page=1", headers)
    link = first.headers.get("Link", "")
    last_page = re.search(r"[?&]page=(\d+)>; rel=\"last\"", link)
    return int(last_page.group(1)) if last_page else len(first.json())


def _file_statistics(tree: list[dict[str, Any]]) -> dict[str, Any]:
    """Create precise, reusable structural counts from a GitHub recursive tree."""
    files = [item["path"] for item in tree if item.get("type") == "blob"]
    extensions: dict[str, int] = {}
    for file_path in files:
        extension = Path(file_path).suffix.lower() or "[no extension]"
        extensions[extension] = extensions.get(extension, 0) + 1

    def is_test_file(file_path: str) -> bool:
        path = file_path.lower()
        name = Path(path).name
        return (
            "/tests/" in f"/{path}"
            or "/test/" in f"/{path}"
            or name.startswith("test_")
            or name.endswith("_test.py")
            or name in {"test.py", "tests.py"}
        )

    top_level_directories = sorted(
        item["path"]
        for item in tree
        if item.get("type") == "tree" and "/" not in item["path"]
    )
    return {
        "total_files": len(files),
        "by_extension": dict(sorted(extensions.items())),
        "python_files": extensions.get(".py", 0),
        "test_files": sum(is_test_file(file_path) for file_path in files),
        "top_level_directories": top_level_directories,
    }


def analyze_repo(github_url: str, github_token: str | None = None) -> dict[str, Any]:
    """Clone a public/private GitHub repository and collect useful metadata.

    Errors are returned in the result instead of being raised so callers can show
    a friendly message for malformed URLs, inaccessible private repositories, and
    GitHub API failures.
    """
    try:
        owner, repo_name = _parse_github_url(github_url)
        headers = _headers(github_token)
        api_base = f"{GITHUB_API}/repos/{owner}/{repo_name}"
        repo_info = _request(api_base, headers).json()

        languages = _request(f"{api_base}/languages", headers).json()
        contributors = _all_contributors(api_base, headers)
        commit_count = _commit_count(api_base, headers)

        default_branch = repo_info.get("default_branch", "main")
        readme_response = requests.get(
            f"{api_base}/readme", headers={**headers, "Accept": "application/vnd.github.raw+json"}, timeout=30
        )
        readme = readme_response.text if readme_response.ok else ""

        tree = _request(f"{api_base}/git/trees/{default_branch}?recursive=1", headers).json().get("tree", [])
        file_tree = [item["path"] for item in tree if item.get("type") == "blob"]
        file_statistics = _file_statistics(tree)

        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        safe_name = re.sub(r"[^A-Za-z0-9_.-]", "-", f"{owner}-{repo_name}")
        repo_path = data_dir / safe_name
        if not repo_path.exists():
            clone_url = repo_info.get("clone_url", github_url)
            # The API token also makes cloning an accessible private repository work.
            # It is never stored in the returned metadata or written to a remote URL.
            if github_token and not github_token.startswith("your_") and clone_url.startswith("https://"):
                clone_url = clone_url.replace("https://", f"https://x-access-token:{quote(github_token, safe='')}@", 1)
            Repo.clone_from(clone_url, repo_path, depth=1)

        return {
            "success": True,
            "repo_name": repo_info.get("full_name", f"{owner}/{repo_name}"),
            "repo_path": str(repo_path.resolve()),
            "languages": languages,
            "contributors": contributors,
            "commit_count": commit_count,
            "readme": readme,
            "file_tree": file_tree,
            "file_statistics": file_statistics,
        }
    except Exception as error:
        # GitPython exceptions and API errors share the same UI-friendly contract.
        message = str(error)
        if github_token:
            message = message.replace(github_token, "***")
        return {"success": False, "error": message}
