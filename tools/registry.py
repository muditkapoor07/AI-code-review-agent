"""Tool registry: maps Claude tool names to Python callables + JSON schemas."""

from __future__ import annotations

from functools import partial
from typing import Any, Callable

from github.client import GitHubClient
from tools import github_tools, static_analysis_tools, security_tools, code_tools


TOOL_SCHEMAS: list[dict] = [
    {
        "name": "fetch_pr_metadata",
        "description": (
            "Fetch high-level metadata about a GitHub pull request: title, description, "
            "author, branch names, labels, and change statistics. Call this first."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "owner": {"type": "string", "description": "GitHub repository owner"},
                "repo": {"type": "string", "description": "GitHub repository name"},
                "pr_number": {"type": "integer", "description": "Pull request number"},
            },
            "required": ["owner", "repo", "pr_number"],
        },
    },
    {
        "name": "fetch_pr_diff",
        "description": (
            "Fetch the list of changed files and their unified diffs. "
            "Returns filename, status (added/modified/removed), patch hunks, and line counts."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "pr_number": {"type": "integer"},
                "max_files": {
                    "type": "integer",
                    "description": "Maximum files to return (default 50)",
                    "default": 50,
                },
            },
            "required": ["owner", "repo", "pr_number"],
        },
    },
    {
        "name": "fetch_file_content",
        "description": (
            "Fetch the full content of a specific file at the PR head ref. "
            "Use when the diff patch is insufficient, or to analyze the complete file."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "ref": {"type": "string", "description": "Git ref (commit SHA or branch name)"},
                "filepath": {"type": "string", "description": "Full file path relative to repo root"},
            },
            "required": ["owner", "repo", "ref", "filepath"],
        },
    },
    {
        "name": "fetch_pr_commits",
        "description": (
            "Fetch the list of commits in the pull request with messages and authors. "
            "Useful for understanding intent and spotting fixup/churn patterns."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "pr_number": {"type": "integer"},
            },
            "required": ["owner", "repo", "pr_number"],
        },
    },
    {
        "name": "analyze_complexity",
        "description": (
            "Run cyclomatic complexity and maintainability index analysis on Python source "
            "using radon. Returns per-function complexity ranks (A=best, F=worst) and MI score."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "source_code": {"type": "string", "description": "Full Python source code"},
                "filename": {"type": "string"},
            },
            "required": ["source_code", "filename"],
        },
    },
    {
        "name": "analyze_syntax",
        "description": "Check Python source code for syntax errors using the ast module.",
        "input_schema": {
            "type": "object",
            "properties": {
                "source_code": {"type": "string"},
                "filename": {"type": "string"},
            },
            "required": ["source_code", "filename"],
        },
    },
    {
        "name": "count_code_metrics",
        "description": (
            "Compute raw code metrics: LOC, SLOC, comment lines, blank lines, "
            "and comment ratio using radon."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "source_code": {"type": "string"},
                "filename": {"type": "string"},
            },
            "required": ["source_code", "filename"],
        },
    },
    {
        "name": "run_bandit_scan",
        "description": (
            "Run Bandit SAST on Python source code to detect OWASP vulnerabilities: "
            "injection, hardcoded secrets, unsafe deserialization, weak crypto. "
            "Returns findings with CWE IDs and severity. Use on all Python files touching "
            "auth, crypto, I/O, network, or user input."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "source_code": {"type": "string", "description": "Python source code to scan"},
                "filename": {"type": "string"},
            },
            "required": ["source_code", "filename"],
        },
    },
    {
        "name": "run_dependency_audit",
        "description": (
            "Run pip-audit on requirements file content to find known CVEs in dependencies. "
            "Returns vulnerable packages with CVE IDs and fix versions. "
            "Use whenever requirements.txt or pyproject.toml is changed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "requirements_content": {
                    "type": "string",
                    "description": "Raw text content of requirements.txt or pip requirements file",
                },
                "source_filename": {
                    "type": "string",
                    "description": "Original filename for logging",
                },
            },
            "required": ["requirements_content"],
        },
    },
    {
        "name": "extract_functions",
        "description": (
            "Extract function and class definitions with line ranges from source code. "
            "Use to map structure before deep analysis."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "source_code": {"type": "string"},
                "filename": {"type": "string"},
                "language": {
                    "type": "string",
                    "enum": ["python", "javascript", "typescript", "go", "java", "other"],
                },
            },
            "required": ["source_code", "filename", "language"],
        },
    },
    {
        "name": "search_patterns",
        "description": (
            "Search source code for specific regex patterns. "
            "Use to hunt for anti-patterns: SQL string concat, eval(), hardcoded IPs, "
            "TODO/FIXME markers, print debugging, os.system calls, etc."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "source_code": {"type": "string"},
                "filename": {"type": "string"},
                "patterns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of Python regex patterns to search for",
                },
            },
            "required": ["source_code", "filename", "patterns"],
        },
    },
    {
        "name": "detect_redundant_code",
        "description": (
            "Detect redundant and dead code: unused imports, duplicate function/class definitions, "
            "unreachable statements after return/raise/break/continue, and TODO stubs. "
            "Always call this during snippet reviews and on key changed files in PRs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "source_code": {"type": "string", "description": "Source code to analyse"},
                "filename": {"type": "string"},
                "language": {
                    "type": "string",
                    "enum": ["python", "javascript", "typescript", "go", "java", "other"],
                },
            },
            "required": ["source_code", "filename", "language"],
        },
    },
    {
        "name": "detect_bugs",
        "description": (
            "Detect common programming bugs: mutable default arguments, bare except clauses, "
            "silent exception swallowing, '== None' instead of 'is None', shadowed builtins, "
            "assert used for runtime validation, identity-vs-equality misuse ('is' with literals). "
            "For JavaScript/TypeScript: loose null checks, empty catch blocks, eval(), innerHTML, debugger. "
            "Always call this during snippet reviews."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "source_code": {"type": "string"},
                "filename": {"type": "string"},
                "language": {
                    "type": "string",
                    "enum": ["python", "javascript", "typescript", "go", "java", "other"],
                },
            },
            "required": ["source_code", "filename", "language"],
        },
    },
]


def _github_unavailable(*args, **kwargs) -> dict:
    return {
        "success": False,
        "error": "GitHub tools are not available in snippet review mode.",
        "result": {},
    }


class ToolRegistry:
    def __init__(self, github_client: GitHubClient | None) -> None:
        self._gh = github_client

        if github_client is not None:
            gh_tools: dict[str, Callable[..., dict]] = {
                "fetch_pr_metadata": partial(github_tools.fetch_pr_metadata, github_client),
                "fetch_pr_diff": partial(github_tools.fetch_pr_diff, github_client),
                "fetch_file_content": partial(github_tools.fetch_file_content, github_client),
                "fetch_pr_commits": partial(github_tools.fetch_pr_commits, github_client),
            }
        else:
            gh_tools = {
                "fetch_pr_metadata": _github_unavailable,
                "fetch_pr_diff": _github_unavailable,
                "fetch_file_content": _github_unavailable,
                "fetch_pr_commits": _github_unavailable,
            }

        self._tools: dict[str, Callable[..., dict]] = {
            **gh_tools,
            "analyze_complexity": static_analysis_tools.analyze_complexity,
            "analyze_syntax": static_analysis_tools.analyze_syntax,
            "count_code_metrics": static_analysis_tools.count_code_metrics,
            "run_bandit_scan": security_tools.run_bandit_scan,
            "run_dependency_audit": security_tools.run_dependency_audit,
            "extract_functions": code_tools.extract_functions,
            "search_patterns": code_tools.search_patterns,
            "detect_redundant_code": code_tools.detect_redundant_code,
            "detect_bugs": code_tools.detect_bugs,
        }

    def get_tool_schemas(self) -> list[dict]:
        """Anthropic-format schemas (kept for reference)."""
        return TOOL_SCHEMAS

    def get_openai_tool_schemas(self) -> list[dict]:
        """OpenAI/Groq-format tool schemas."""
        return [
            {
                "type": "function",
                "function": {
                    "name": s["name"],
                    "description": s["description"],
                    "parameters": s["input_schema"],
                },
            }
            for s in TOOL_SCHEMAS
        ]

    # Tools that accept raw source code — cap to avoid Groq hallucinating truncated strings
    _SOURCE_CODE_TOOLS = {"analyze_complexity", "analyze_syntax", "count_code_metrics",
                          "run_bandit_scan", "extract_functions", "search_patterns",
                          "detect_redundant_code", "detect_bugs"}
    _MAX_SOURCE_CHARS = 6000

    def execute(self, name: str, **kwargs: Any) -> dict:
        tool_fn = self._tools.get(name)
        if tool_fn is None:
            return {
                "success": False,
                "error": f"Unknown tool: {name!r}",
                "result": {},
            }

        # Truncate oversized source_code to prevent malformed JSON from model
        if name in self._SOURCE_CODE_TOOLS and "source_code" in kwargs:
            src = kwargs["source_code"]
            if isinstance(src, str) and len(src) > self._MAX_SOURCE_CHARS:
                kwargs["source_code"] = src[:self._MAX_SOURCE_CHARS] + "\n# ... [truncated]"

        try:
            return tool_fn(**kwargs)
        except TypeError as exc:
            return {
                "success": False,
                "error": f"Invalid arguments for {name!r}: {exc}",
                "result": {},
            }
        except Exception as exc:
            return {
                "success": False,
                "error": f"Tool {name!r} raised: {exc}",
                "result": {},
            }
