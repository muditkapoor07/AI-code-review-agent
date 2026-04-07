"""GitHub-facing tools exposed to the Claude agent."""

from __future__ import annotations

import os
from typing import Any

from github.client import GitHubClient

_PATCH_LINE_LIMIT = 50
_LANGUAGE_MAP = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".tsx": "typescript", ".jsx": "javascript", ".go": "go",
    ".java": "java", ".rb": "ruby", ".rs": "rust", ".cs": "csharp",
    ".cpp": "cpp", ".c": "c", ".php": "php", ".swift": "swift",
    ".kt": "kotlin", ".sh": "bash", ".yaml": "yaml", ".yml": "yaml",
    ".json": "json", ".toml": "toml", ".md": "markdown",
    ".html": "html", ".css": "css", ".sql": "sql",
}


def _detect_language(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    return _LANGUAGE_MAP.get(ext, "unknown")


def _truncate_patch(patch: str | None) -> tuple[str, bool]:
    if not patch:
        return "", False
    lines = patch.splitlines()
    if len(lines) <= _PATCH_LINE_LIMIT:
        return patch, False
    truncated = "\n".join(lines[:_PATCH_LINE_LIMIT])
    truncated += f"\n[TRUNCATED — {len(lines) - _PATCH_LINE_LIMIT} lines omitted]"
    return truncated, True


def fetch_pr_metadata(client: GitHubClient, owner: str, repo: str, pr_number: int) -> dict[str, Any]:
    try:
        pr = client.get_pull_request(owner, repo, pr_number)
        return {
            "success": True,
            "result": {
                "title": pr["title"],
                "description": pr.get("body") or "",
                "author": pr["user"]["login"],
                "state": pr["state"],
                "base_branch": pr["base"]["ref"],
                "head_branch": pr["head"]["ref"],
                "head_sha": pr["head"]["sha"],
                "created_at": pr["created_at"],
                "updated_at": pr["updated_at"],
                "labels": [lb["name"] for lb in pr.get("labels", [])],
                "draft": pr.get("draft", False),
                "mergeable": pr.get("mergeable"),
                "additions": pr.get("additions", 0),
                "deletions": pr.get("deletions", 0),
                "changed_files": pr.get("changed_files", 0),
                "review_comments": pr.get("review_comments", 0),
                "commits": pr.get("commits", 0),
            },
        }
    except Exception as exc:
        return {"success": False, "error": str(exc), "result": {}}


def fetch_pr_diff(
    client: GitHubClient,
    owner: str,
    repo: str,
    pr_number: int,
    max_files: int = 50,
) -> dict[str, Any]:
    try:
        files = client.get_pr_files(owner, repo, pr_number)[:max_files]
        result_files = []
        for f in files:
            patch, was_truncated = _truncate_patch(f.get("patch"))
            result_files.append({
                "filename": f["filename"],
                "status": f["status"],
                "additions": f.get("additions", 0),
                "deletions": f.get("deletions", 0),
                "language": _detect_language(f["filename"]),
                "patch": patch,
                "patch_truncated": was_truncated,
            })
        return {
            "success": True,
            "result": {
                "files": result_files,
                "total_files": len(files),
                "truncated_to": max_files if len(files) == max_files else None,
            },
        }
    except Exception as exc:
        return {"success": False, "error": str(exc), "result": {}}


_FILE_LINE_LIMIT = 80

def fetch_file_content(
    client: GitHubClient,
    owner: str,
    repo: str,
    ref: str,
    filepath: str,
) -> dict[str, Any]:
    try:
        content = client.get_file_content(owner, repo, ref, filepath)
        lines = content.splitlines()
        truncated = False
        if len(lines) > _FILE_LINE_LIMIT:
            content = "\n".join(lines[:_FILE_LINE_LIMIT]) + f"\n... [{len(lines) - _FILE_LINE_LIMIT} more lines truncated]"
            truncated = True
        return {
            "success": True,
            "result": {
                "filepath": filepath,
                "ref": ref,
                "content": content,
                "line_count": len(lines),
                "truncated": truncated,
                "language": _detect_language(filepath),
            },
        }
    except Exception as exc:
        return {"success": False, "error": str(exc), "result": {}}


def fetch_pr_commits(
    client: GitHubClient,
    owner: str,
    repo: str,
    pr_number: int,
) -> dict[str, Any]:
    try:
        commits = client.get_pr_commits(owner, repo, pr_number)
        return {
            "success": True,
            "result": {
                "commits": [
                    {
                        "sha": c["sha"][:8],
                        "message": c["commit"]["message"],
                        "author": c["commit"]["author"]["name"],
                        "date": c["commit"]["author"]["date"],
                    }
                    for c in commits
                ],
                "count": len(commits),
            },
        }
    except Exception as exc:
        return {"success": False, "error": str(exc), "result": {}}
