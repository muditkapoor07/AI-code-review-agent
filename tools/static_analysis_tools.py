"""Static analysis tools: complexity, syntax, and raw metrics."""

from __future__ import annotations

import ast
from typing import Any


def analyze_syntax(source_code: str, filename: str) -> dict[str, Any]:
    """Check Python source for syntax errors using the ast module."""
    try:
        ast.parse(source_code, filename=filename)
        return {
            "success": True,
            "result": {"valid": True, "errors": [], "filename": filename},
        }
    except SyntaxError as exc:
        return {
            "success": True,
            "result": {
                "valid": False,
                "errors": [f"Line {exc.lineno}: {exc.msg}"],
                "filename": filename,
            },
        }
    except Exception as exc:
        return {"success": False, "error": str(exc), "result": {}}


def analyze_complexity(source_code: str, filename: str) -> dict[str, Any]:
    """Run cyclomatic complexity and maintainability index via radon."""
    try:
        from radon.complexity import cc_visit, ComplexityVisitor  # noqa: F401
        from radon.metrics import mi_visit
        from radon.raw import analyze as raw_analyze
    except ImportError:
        return {
            "success": False,
            "error": "radon is not installed. Run: pip install radon",
            "result": {},
        }

    try:
        blocks = cc_visit(source_code)
        mi_score = mi_visit(source_code, multi=True)

        functions = []
        rank_map = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6}

        for block in blocks:
            rank = block.rank if hasattr(block, "rank") else _complexity_rank(block.complexity)
            functions.append({
                "name": block.name,
                "type": block.type if hasattr(block, "type") else "function",
                "line_number": block.lineno,
                "complexity": block.complexity,
                "rank": rank,
                "rank_order": rank_map.get(rank, 0),
            })

        functions.sort(key=lambda x: x["rank_order"], reverse=True)
        high_complexity = [f for f in functions if f["rank_order"] >= 4]

        avg_complexity = (
            sum(f["complexity"] for f in functions) / len(functions)
            if functions
            else 0
        )

        raw = raw_analyze(source_code)

        return {
            "success": True,
            "result": {
                "filename": filename,
                "functions": functions,
                "high_complexity_functions": high_complexity,
                "average_complexity": round(avg_complexity, 2),
                "maintainability_index": round(float(mi_score), 2),
                "loc": raw.loc,
                "sloc": raw.sloc,
            },
        }
    except Exception as exc:
        return {"success": False, "error": str(exc), "result": {}}


def count_code_metrics(source_code: str, filename: str) -> dict[str, Any]:
    """Compute raw code metrics using radon."""
    try:
        from radon.raw import analyze
    except ImportError:
        return {
            "success": False,
            "error": "radon is not installed. Run: pip install radon",
            "result": {},
        }

    try:
        metrics = analyze(source_code)
        comment_ratio = (
            round(metrics.comments / metrics.loc * 100, 1) if metrics.loc > 0 else 0
        )
        return {
            "success": True,
            "result": {
                "filename": filename,
                "loc": metrics.loc,
                "sloc": metrics.sloc,
                "comments": metrics.comments,
                "multi_line_comments": metrics.multi,
                "blank": metrics.blank,
                "comment_ratio_pct": comment_ratio,
            },
        }
    except Exception as exc:
        return {"success": False, "error": str(exc), "result": {}}


def _complexity_rank(complexity: int) -> str:
    if complexity <= 5:
        return "A"
    if complexity <= 10:
        return "B"
    if complexity <= 15:
        return "C"
    if complexity <= 20:
        return "D"
    if complexity <= 25:
        return "E"
    return "F"
