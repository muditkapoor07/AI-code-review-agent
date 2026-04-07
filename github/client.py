"""Thin httpx wrapper for the GitHub REST API."""

from __future__ import annotations

import base64
import re
from typing import Any

import httpx


class GitHubAPIError(Exception):
    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        super().__init__(f"GitHub API error {status_code}: {message}")


class GitHubClient:
    def __init__(self, token: str, base_url: str = "https://api.github.com") -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            headers={
                "Authorization": f"token {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            timeout=30.0,
        )

    def _get(self, path: str, params: dict | None = None) -> Any:
        url = f"{self._base_url}{path}"
        response = self._client.get(url, params=params)
        if response.status_code >= 400:
            try:
                msg = response.json().get("message", response.text)
            except Exception:
                msg = response.text
            raise GitHubAPIError(response.status_code, msg)
        return response.json()

    def _get_paginated(self, path: str, per_page: int = 100) -> list[dict]:
        results: list[dict] = []
        page = 1
        while True:
            data = self._get(path, params={"per_page": per_page, "page": page})
            if not data:
                break
            results.extend(data)
            if len(data) < per_page:
                break
            page += 1
        return results

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def parse_pr_url(url: str) -> tuple[str, str, int]:
        """Parse a GitHub PR URL into (owner, repo, pr_number)."""
        pattern = r"https?://github\.com/([^/]+)/([^/]+)/pull/(\d+)"
        match = re.match(pattern, url.rstrip("/"))
        if not match:
            raise ValueError(
                f"Invalid GitHub PR URL: {url!r}\n"
                "Expected format: https://github.com/owner/repo/pull/123"
            )
        owner, repo, number = match.groups()
        return owner, repo, int(number)

    def get_pull_request(self, owner: str, repo: str, pr_number: int) -> dict:
        return self._get(f"/repos/{owner}/{repo}/pulls/{pr_number}")

    def get_pr_files(self, owner: str, repo: str, pr_number: int) -> list[dict]:
        return self._get_paginated(f"/repos/{owner}/{repo}/pulls/{pr_number}/files")

    def get_pr_commits(self, owner: str, repo: str, pr_number: int) -> list[dict]:
        return self._get_paginated(f"/repos/{owner}/{repo}/pulls/{pr_number}/commits")

    def get_file_content(self, owner: str, repo: str, ref: str, path: str) -> str:
        """Fetch and decode a file's content at a specific git ref."""
        data = self._get(
            f"/repos/{owner}/{repo}/contents/{path}",
            params={"ref": ref},
        )
        if isinstance(data, list):
            raise ValueError(f"Path {path!r} is a directory, not a file")
        if data.get("encoding") == "base64":
            return base64.b64decode(data["content"]).decode("utf-8", errors="replace")
        return data.get("content", "")

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "GitHubClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
