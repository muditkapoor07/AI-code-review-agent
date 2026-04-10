"""Code structure tools: function extraction, pattern searching, redundancy and bug detection."""

from __future__ import annotations

import ast
import re
from typing import Any

_PYTHON_BUILTINS = {
    "list", "dict", "set", "tuple", "str", "int", "float", "bool", "bytes",
    "type", "object", "map", "filter", "zip", "range", "enumerate",
    "len", "sum", "min", "max", "abs", "round", "sorted", "reversed",
    "print", "input", "open", "super", "property", "staticmethod",
    "classmethod", "id", "hash", "repr", "format", "vars", "dir",
    "getattr", "setattr", "hasattr", "delattr", "isinstance", "issubclass",
    "iter", "next", "all", "any", "bin", "hex", "oct", "ord", "chr",
}


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


def detect_redundant_code(source_code: str, filename: str, language: str) -> dict[str, Any]:
    """Detect unused imports, duplicate definitions, dead code, and unreachable statements."""
    issues: list[dict] = []

    if language == "python":
        try:
            tree = ast.parse(source_code, filename=filename)
        except SyntaxError as exc:
            return {"success": False, "error": f"Syntax error: {exc}", "result": {}}

        # ── Unused imports ──────────────────────────────────────────────
        imported: dict[str, int] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.asname or alias.name.split(".")[0]
                    imported[name] = node.lineno
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    name = alias.asname or alias.name
                    imported[name] = node.lineno

        used: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            if isinstance(node, ast.Name):
                used.add(node.id)
            elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                used.add(node.value.id)

        for name, lineno in imported.items():
            if name not in used:
                issues.append({
                    "type": "unused_import",
                    "line": lineno,
                    "message": f"'{name}' is imported but never used — remove to reduce noise",
                })

        # ── Duplicate function / class definitions ──────────────────────
        defined: dict[str, int] = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name in defined:
                    issues.append({
                        "type": "duplicate_definition",
                        "line": node.lineno,
                        "message": (
                            f"'{node.name}' re-defined at line {node.lineno} "
                            f"(first definition at line {defined[node.name]}) — "
                            "the earlier version is silently overwritten"
                        ),
                    })
                else:
                    defined[node.name] = node.lineno

        # ── Dead code after return / raise / break / continue ───────────
        def _check_dead(body: list) -> None:
            for i, stmt in enumerate(body[:-1]):
                if isinstance(stmt, (ast.Return, ast.Raise)):
                    keyword = "return" if isinstance(stmt, ast.Return) else "raise"
                    nxt = body[i + 1]
                    issues.append({
                        "type": "dead_code",
                        "line": getattr(nxt, "lineno", "?"),
                        "message": (
                            f"Unreachable statement at line {getattr(nxt, 'lineno', '?')} "
                            f"— control never reaches here after '{keyword}' at line {stmt.lineno}"
                        ),
                    })
                if isinstance(stmt, (ast.Break, ast.Continue)):
                    keyword = "break" if isinstance(stmt, ast.Break) else "continue"
                    nxt = body[i + 1]
                    issues.append({
                        "type": "dead_code",
                        "line": getattr(nxt, "lineno", "?"),
                        "message": (
                            f"Unreachable statement at line {getattr(nxt, 'lineno', '?')} "
                            f"— after '{keyword}' at line {stmt.lineno}"
                        ),
                    })

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                  ast.For, ast.While, ast.If, ast.With)):
                if hasattr(node, "body"):
                    _check_dead(node.body)
                if hasattr(node, "orelse") and node.orelse:
                    _check_dead(node.orelse)

    else:
        # ── Regex-based for non-Python ──────────────────────────────────
        lines = source_code.splitlines()
        func_re = re.compile(r"(?:function|def|func)\s+(\w+)\s*\(")
        todo_re = re.compile(r"//\s*TODO|//\s*FIXME|#\s*TODO|#\s*FIXME|/\*\s*TODO", re.IGNORECASE)
        defined_regex: dict[str, int] = {}

        for i, line in enumerate(lines, 1):
            m = func_re.search(line)
            if m:
                name = m.group(1)
                if name in defined_regex:
                    issues.append({
                        "type": "duplicate_definition",
                        "line": i,
                        "message": f"'{name}' defined again at line {i} (first at line {defined_regex[name]})",
                    })
                else:
                    defined_regex[name] = i
            if todo_re.search(line):
                issues.append({
                    "type": "todo_stub",
                    "line": i,
                    "message": f"TODO/FIXME marker — unfinished or placeholder code",
                })

    categories = list({iss["type"] for iss in issues})
    return {
        "success": True,
        "result": {
            "filename": filename,
            "language": language,
            "issue_count": len(issues),
            "categories": categories,
            "issues": issues[:40],
        },
    }


def detect_bugs(source_code: str, filename: str, language: str) -> dict[str, Any]:
    """Detect common programming bugs: mutable defaults, bare excepts, None comparisons,
    shadowed builtins, silent exception swallowing, and identity-vs-equality misuse."""
    bugs: list[dict] = []

    if language == "python":
        try:
            tree = ast.parse(source_code, filename=filename)
        except SyntaxError as exc:
            return {"success": False, "error": f"Syntax error: {exc}", "result": {}}

        for node in ast.walk(tree):

            # ── Mutable default argument ────────────────────────────────
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                all_defaults = node.args.defaults + [
                    d for d in node.args.kw_defaults if d is not None
                ]
                for default in all_defaults:
                    if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                        kind = type(default).__name__.replace("ast.", "").lower()
                        bugs.append({
                            "type": "mutable_default_arg",
                            "severity": "high",
                            "line": node.lineno,
                            "message": (
                                f"'{node.name}' uses a mutable {kind} as a default argument — "
                                "it is shared across ALL calls; state leaks between invocations"
                            ),
                        })

            # ── Bare except ─────────────────────────────────────────────
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                bugs.append({
                    "type": "bare_except",
                    "severity": "medium",
                    "line": node.lineno,
                    "message": (
                        "Bare 'except:' catches SystemExit, KeyboardInterrupt, and "
                        "GeneratorExit — use 'except Exception:' at minimum"
                    ),
                })

            # ── Silent exception swallow ─────────────────────────────────
            if isinstance(node, ast.ExceptHandler):
                non_docstring = [s for s in node.body if not (
                    isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)
                )]
                if all(isinstance(s, ast.Pass) for s in non_docstring) or not non_docstring:
                    bugs.append({
                        "type": "silent_exception",
                        "severity": "high",
                        "line": node.lineno,
                        "message": "Exception caught and silently swallowed — errors hidden, debugging becomes very difficult",
                    })

            # ── == None / != None ────────────────────────────────────────
            if isinstance(node, ast.Compare):
                for op, comp in zip(node.ops, node.comparators):
                    if (isinstance(op, (ast.Eq, ast.NotEq))
                            and isinstance(comp, ast.Constant)
                            and comp.value is None):
                        op_str = "==" if isinstance(op, ast.Eq) else "!="
                        bugs.append({
                            "type": "none_comparison",
                            "severity": "low",
                            "line": node.lineno,
                            "message": (
                                f"'{op_str} None' should be "
                                f"'{'is' if isinstance(op, ast.Eq) else 'is not'} None' — "
                                "PEP 8 and correct identity semantics"
                            ),
                        })

                    # ── 'is' used with non-None literal ─────────────────
                    if (isinstance(op, (ast.Is, ast.IsNot))
                            and isinstance(comp, ast.Constant)
                            and comp.value is not None
                            and not isinstance(comp.value, bool)):
                        bugs.append({
                            "type": "is_literal_comparison",
                            "severity": "medium",
                            "line": node.lineno,
                            "message": (
                                f"'is' used to compare with literal {comp.value!r} — "
                                "use '==' instead; 'is' tests identity not equality "
                                "(may work by coincidence via interning but not guaranteed)"
                            ),
                        })

            # ── Shadowed builtins ────────────────────────────────────────
            if isinstance(node, (ast.Assign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name) and target.id in _PYTHON_BUILTINS:
                        bugs.append({
                            "type": "shadowed_builtin",
                            "severity": "medium",
                            "line": node.lineno,
                            "message": (
                                f"'{target.id}' shadows a Python builtin — "
                                "any code that later calls the real builtin in this scope will fail"
                            ),
                        })

            # ── assert used for runtime validation ───────────────────────
            if isinstance(node, ast.Assert):
                bugs.append({
                    "type": "assert_for_validation",
                    "severity": "low",
                    "line": node.lineno,
                    "message": (
                        "'assert' is stripped when Python runs with -O or -OO — "
                        "use explicit 'if ... raise ValueError(...)' for runtime guards"
                    ),
                })

    else:
        # ── Regex-based for non-Python ──────────────────────────────────
        lines = source_code.splitlines()
        patterns = [
            (r"==\s*null\b",             "medium", "Use '=== null' (strict equality) instead of '== null'"),
            (r"!=\s*null\b",             "medium", "Use '!== null' (strict inequality) instead of '!= null'"),
            (r"==\s*undefined\b",        "medium", "Use '=== undefined' instead of '== undefined'"),
            (r"catch\s*\([^)]*\)\s*\{\s*\}", "high", "Empty catch block — exception silently swallowed"),
            (r"\bconsole\.log\s*\(",      "low",    "Debug console.log() left in production code"),
            (r"\bdebugger\s*;",           "high",   "'debugger' statement left in code — halts execution in dev tools"),
            (r"\beval\s*\(",              "high",   "eval() executes arbitrary code — severe security risk"),
            (r"document\.write\s*\(",     "high",   "document.write() overwrites entire document — use DOM methods"),
            (r"innerHTML\s*=\s*[^\"'`]",  "medium", "Unescaped assignment to innerHTML — potential XSS"),
            (r"TODO|FIXME|HACK|XXX",      "low",    "Unresolved TODO/FIXME/HACK marker"),
        ]
        for i, line in enumerate(lines, 1):
            for pattern, severity, message in patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    bugs.append({
                        "type": "bug_pattern",
                        "severity": severity,
                        "line": i,
                        "message": message,
                        "line_content": line.strip()[:120],
                    })

    high   = sum(1 for b in bugs if b.get("severity") == "high")
    medium = sum(1 for b in bugs if b.get("severity") == "medium")
    low    = sum(1 for b in bugs if b.get("severity") == "low")

    return {
        "success": True,
        "result": {
            "filename": filename,
            "language": language,
            "bug_count": len(bugs),
            "severity_breakdown": {"high": high, "medium": medium, "low": low},
            "bugs": bugs[:40],
        },
    }
