"""Code structure tools: function extraction and pattern searching."""

from __future__ import annotations

import ast
import re
from typing import Any


def extract_functions(source_code: str, filename: str, language: str) -> dict[str, Any]:
    """Extract function and class definitions with their line ranges."""
    if language == "python":
        return _extract_python(source_code, filename)
    return _extract_regex(source_code, filename, language)


def _extract_python(source_code: str, filename: str) -> dict[str, Any]:
    try:
        tree = ast.parse(source_code, filename=filename)
    except SyntaxError as exc:
        return {"success": False, "error": f"Syntax error: {exc}", "result": {}}

    items = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end_line = getattr(node, "end_lineno", None)
            args = [a.arg for a in node.args.args]
            items.append({
                "type": "async_function" if isinstance(node, ast.AsyncFunctionDef) else "function",
                "name": node.name,
                "line_start": node.lineno,
                "line_end": end_line,
                "args": args,
                "decorators": [
                    ast.unparse(d) if hasattr(ast, "unparse") else str(d)
                    for d in node.decorator_list
                ],
            })
        elif isinstance(node, ast.ClassDef):
            end_line = getattr(node, "end_lineno", None)
            items.append({
                "type": "class",
                "name": node.name,
                "line_start": node.lineno,
                "line_end": end_line,
                "bases": [
                    ast.unparse(b) if hasattr(ast, "unparse") else str(b)
                    for b in node.bases
                ],
                "decorators": [
                    ast.unparse(d) if hasattr(ast, "unparse") else str(d)
                    for d in node.decorator_list
                ],
            })

    items.sort(key=lambda x: x["line_start"])
    return {
        "success": True,
        "result": {
            "filename": filename,
            "language": "python",
            "definitions": items,
            "count": len(items),
        },
    }


_REGEX_PATTERNS: dict[str, str] = {
    "javascript": r"(?:^|\s)(?:async\s+)?function\s+(\w+)\s*\(|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\(",
    "typescript": r"(?:^|\s)(?:async\s+)?function\s+(\w+)\s*\(|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\(|^\s*(?:public|private|protected|static|async|\s)*(\w+)\s*\(",
    "go": r"^func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\(",
    "java": r"(?:public|private|protected|static|final|\s)+\w+\s+(\w+)\s*\(",
    "other": r"(?:function|def|func)\s+(\w+)\s*\(",
}


def _extract_regex(source_code: str, filename: str, language: str) -> dict[str, Any]:
    pattern = _REGEX_PATTERNS.get(language, _REGEX_PATTERNS["other"])
    items = []
    for i, line in enumerate(source_code.splitlines(), start=1):
        match = re.search(pattern, line, re.MULTILINE)
        if match:
            name = next((g for g in match.groups() if g), "<anonymous>")
            items.append({"type": "function", "name": name, "line_start": i, "line_end": None})
    return {
        "success": True,
        "result": {
            "filename": filename,
            "language": language,
            "definitions": items,
            "count": len(items),
            "note": "Line ends unavailable for non-Python files",
        },
    }


def search_patterns(
    source_code: str,
    filename: str,
    patterns: list[str],
) -> dict[str, Any]:
    """Regex search across source code for specific anti-patterns."""
    results = []
    lines = source_code.splitlines()

    for pattern in patterns:
        matches = []
        try:
            compiled = re.compile(pattern)
            for i, line in enumerate(lines, start=1):
                if compiled.search(line):
                    matches.append({"line_number": i, "line_content": line.rstrip()})
        except re.error as exc:
            matches = [{"error": f"Invalid regex: {exc}"}]

        results.append({
            "pattern": pattern,
            "match_count": len(matches),
            "matches": matches[:50],  # cap to avoid flooding context
        })

    total_matches = sum(r["match_count"] for r in results)
    return {
        "success": True,
        "result": {
            "filename": filename,
            "patterns_searched": len(patterns),
            "total_matches": total_matches,
            "results": results,
        },
    }
