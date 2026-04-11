"""Core agent loop using Groq (OpenAI-compatible API)."""

from __future__ import annotations

import json
import re
import time
from typing import Any, Callable

from openai import OpenAI

from agent.prompts import SYSTEM_PROMPT, INITIAL_USER_MESSAGE, SNIPPET_USER_MESSAGE
from agent.schemas import FinalReview, ReviewScores, ToolUsageEntry
from github.client import GitHubClient
from tools.registry import ToolRegistry
from utils.logger import ReviewLogger


_DEFAULT_MODEL = "llama-3.3-70b-versatile"
_DEFAULT_MAX_PASSES = 10
_MAX_TOKENS = 2048          # per-turn limit during investigation
_MAX_TOKENS_FINAL = 4096    # higher limit for final JSON output
_MAX_TOOL_RESULT_CHARS = 1500
_MAX_HISTORY_MESSAGES = 12
_GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class AgentLoopError(Exception):
    pass


class CodeReviewAgent:
    def __init__(
        self,
        groq_client: OpenAI,
        github_client: GitHubClient | None,
        tool_registry: ToolRegistry,
        logger: ReviewLogger,
        model: str = _DEFAULT_MODEL,
        max_passes: int = _DEFAULT_MAX_PASSES,
        event_callback: Callable[[dict], None] | None = None,
    ) -> None:
        self._client = groq_client
        self._gh = github_client
        self._registry = tool_registry
        self._logger = logger
        self._model = model
        self._max_passes = max_passes
        self._emit = event_callback or (lambda e: None)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def run(self, pr_url: str) -> FinalReview:
        owner, repo, pr_number = GitHubClient.parse_pr_url(pr_url)
        self._emit({"type": "status", "message": f"Reviewing PR #{pr_number} in {owner}/{repo}"})

        initial_message = INITIAL_USER_MESSAGE.format(
            pr_url=pr_url, owner=owner, repo=repo, pr_number=pr_number,
        )
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": initial_message},
        ]
        raw_output, tool_log, pass_count = self._agentic_loop(messages)
        return self._parse_final_review(raw_output, pr_url, tool_log, pass_count)

    def run_snippet(self, code: str, language: str, filename: str) -> FinalReview:
        self._emit({"type": "status", "message": f"Reviewing {filename} ({language})"})

        initial_message = SNIPPET_USER_MESSAGE.format(
            filename=filename, language=language, code=code,
        )
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": initial_message},
        ]
        raw_output, tool_log, pass_count = self._agentic_loop(messages, source_ref=f"snippet:{filename}")
        return self._parse_final_review(raw_output, f"snippet:{filename}", tool_log, pass_count)

    # ------------------------------------------------------------------ #
    # Agent loop                                                           #
    # ------------------------------------------------------------------ #

    def _agentic_loop(self, messages: list[dict], source_ref: str = "") -> tuple[str, list[dict], int]:
        """Returns (raw_output, tool_log, pass_count).
        tool_log is built by the agent — never ask the LLM to reproduce it.
        """
        pass_number = 0
        tool_log: list[dict] = []
        tools = self._registry.get_openai_tool_schemas()

        while pass_number < self._max_passes:
            pass_number += 1
            self._logger.log_pass_start(pass_number)
            self._emit({"type": "pass_start", "pass": pass_number})

            try:
                response = self._call_groq(messages, tools)
            except Exception as api_err:
                err_str = str(api_err)
                if "tool_use_failed" in err_str or "'code': 400" in err_str or "Error code: 400" in err_str:
                    self._emit({"type": "status", "message": "Tool call failed — recovering and synthesizing…"})
                    return self._force_final_answer(messages), tool_log, pass_number
                if "429" in err_str or "rate_limit_exceeded" in err_str:
                    import re as _re
                    wait = _re.search(r'try again in ([^.]+)', err_str)
                    wait_msg = f"Rate limit hit — try again in {wait.group(1)}" if wait else "Rate limit hit — please wait before retrying"
                    self._emit({"type": "error", "message": wait_msg})
                    raise
                raise

            choice = response.choices[0]
            msg = choice.message
            stop_reason = choice.finish_reason

            self._logger.log_pass_end(pass_number, stop_reason)
            self._emit({"type": "pass_end", "pass": pass_number, "stop_reason": stop_reason})

            if msg.content and msg.content.strip():
                self._emit({"type": "thinking", "text": msg.content[:500]})

            if stop_reason == "stop" or not msg.tool_calls:
                text = msg.content or ""
                if "<final_review>" not in text:
                    messages.append({"role": "assistant", "content": text})
                    return self._force_final_answer(messages), tool_log, pass_number
                return text, tool_log, pass_number

            if stop_reason == "tool_calls" or msg.tool_calls:
                msg_dict = msg.model_dump(exclude_unset=True, exclude_none=True)
                msg_dict.pop("annotations", None)
                msg_dict.pop("audio", None)
                msg_dict.pop("refusal", None)
                messages.append(msg_dict)

                for tc in msg.tool_calls:
                    tool_name = tc.function.name
                    try:
                        tool_input = json.loads(tc.function.arguments)
                    except (json.JSONDecodeError, Exception):
                        tool_input = {}
                        self._emit({"type": "tool_end", "tool": tool_name, "success": False,
                                    "summary": "Skipped — malformed arguments", "duration_ms": 0})
                        messages.append({"role": "tool", "tool_call_id": tc.id,
                                         "content": '{"success":false,"error":"malformed arguments","result":{}}'})
                        continue

                    self._emit({"type": "tool_start", "tool": tool_name, "inputs": _sanitize(tool_input)})

                    start = time.monotonic()
                    result = self._registry.execute(tool_name, **tool_input)
                    duration_ms = (time.monotonic() - start) * 1000

                    self._logger.log_tool_call(tool_name, tool_input, result, duration_ms)

                    summary = _result_summary(tool_name, result)
                    self._emit({
                        "type": "tool_end",
                        "tool": tool_name,
                        "success": result.get("success", True),
                        "summary": summary,
                        "duration_ms": round(duration_ms),
                    })

                    # Track tool usage ourselves — don't rely on LLM to reproduce this
                    tool_log.append({
                        "tool": tool_name,
                        "purpose": _tool_purpose(tool_name),
                        "key_finding": summary,
                    })

                    result_str = json.dumps(result, default=str)
                    if len(result_str) > _MAX_TOOL_RESULT_CHARS:
                        result_str = result_str[:_MAX_TOOL_RESULT_CHARS] + '... [truncated]"}'
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_str,
                    })
                continue

            break

        return self._force_final_answer(messages), tool_log, pass_number

    def _call_groq(self, messages: list[dict], tools: list[dict]):
        system_msgs = [m for m in messages if m.get("role") == "system"]
        other_msgs  = [m for m in messages if m.get("role") != "system"]
        trimmed = system_msgs + other_msgs[-_MAX_HISTORY_MESSAGES:]
        return self._client.chat.completions.create(
            model=self._model,
            max_tokens=_MAX_TOKENS,
            tools=tools,
            tool_choice="auto",
            messages=trimmed,
        )

    def _force_final_answer(self, messages: list[dict]) -> str:
        self._emit({"type": "status", "message": "Synthesizing final review…"})
        messages.append({
            "role": "user",
            "content": (
                "Output the final review JSON inside <final_review>...</final_review> tags. "
                "Output NOTHING else — no text before or after the tags.\n\n"
                "IMPORTANT: Do NOT include a 'tool_usage_log' field — it is handled separately.\n\n"
                "Required JSON (keep it concise — max 5 findings):\n"
                '{"pr_url":"","pr_title":"","executive_summary":"1-2 sentences",'
                '"passes_completed":1,'
                '"findings":[{"id":"F001","category":"bug|security|performance|code_quality",'
                '"severity":"critical|high|medium|low|info","file":"","title":"",'
                '"description":"","fix":{"description":"","code":""},"references":[]}],'
                '"scores":{"code_quality":5,"security":5,"performance":5,"overall":5,'
                '"rationale":{"code_quality":"","security":"","performance":"","overall":""}},'
                '"verdict":"APPROVE|REQUEST_CHANGES|REJECT",'
                '"blocking_issues":["plain strings only"]}'
            ),
        })
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=_MAX_TOKENS_FINAL,
            messages=messages[-_MAX_HISTORY_MESSAGES:],
        )
        return response.choices[0].message.content or ""

    # ------------------------------------------------------------------ #
    # Parsing                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_final_review(
        raw_output: str,
        source_ref: str,
        tool_log: list[dict],
        pass_count: int,
    ) -> FinalReview:

        # ── Step 1: extract the JSON string ────────────────────────────
        json_str = _extract_json_str(raw_output)

        # ── Step 2: parse with progressive recovery ─────────────────────
        data = _parse_with_recovery(json_str)

        # ── Step 3: if we still have nothing, nuclear regex extraction ──
        if data is None:
            data = _nuclear_extract(raw_output)

        # ── Step 4: fill mandatory fields from our own tracking ─────────
        if not isinstance(data, dict):
            data = {}

        data.setdefault("pr_url", source_ref)
        data.setdefault("pr_title", "")
        data.setdefault("executive_summary", "Review completed.")
        data["passes_completed"] = pass_count
        # Always use our own tool log — more reliable than LLM output
        data["tool_usage_log"] = tool_log
        data.setdefault("findings", [])
        data.setdefault("scores", {})
        data.setdefault("verdict", "REQUEST_CHANGES")
        data.setdefault("blocking_issues", [])

        # ── Step 5: normalise and coerce ────────────────────────────────
        _strip_problematic_fields(data)

        # ── Step 6: Pydantic validate (validators handle remaining coercion)
        try:
            return FinalReview.model_validate(data)
        except Exception as exc:
            # Absolute last resort — strip findings entirely and return minimal report
            data["findings"] = []
            data["blocking_issues"] = []
            data["scores"] = {"code_quality": 5, "security": 5,
                              "performance": 5, "overall": 5, "rationale": {}}
            try:
                return FinalReview.model_validate(data)
            except Exception as exc2:
                raise AgentLoopError(
                    f"Could not build a valid review even with fallback: {exc2}"
                )


# ------------------------------------------------------------------ #
# JSON extraction & repair helpers                                    #
# ------------------------------------------------------------------ #

def _extract_json_str(raw: str) -> str:
    """Extract the JSON string from raw LLM output."""
    # Try <final_review> tags first
    match = re.search(r"<final_review>(.*?)</final_review>", raw, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
    else:
        # Fallback: find the largest {...} block containing "verdict"
        json_match = re.search(r"\{[\s\S]*\"verdict\"[\s\S]*\}", raw)
        json_str = json_match.group(0) if json_match else raw.strip()

    # Strip markdown fences
    json_str = re.sub(r"^```(?:json)?\s*", "", json_str.strip())
    json_str = re.sub(r"\s*```$", "", json_str)

    # Unwrap list wrapper at string level: [{...}] → {...}
    stripped = json_str.strip()
    if stripped.startswith("["):
        inner = re.search(r"\{[\s\S]*\}", stripped)
        if inner:
            json_str = inner.group(0)

    return json_str


def _parse_with_recovery(json_str: str) -> dict | None:
    """Try to parse JSON with up to 3 recovery attempts. Returns dict or None."""
    # Attempt 1: direct parse
    try:
        data = json.loads(json_str)
        return data[0] if isinstance(data, list) and data else data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass

    # Attempt 2: repair common mistakes
    repaired = _repair_json(json_str)
    try:
        data = json.loads(repaired)
        return data[0] if isinstance(data, list) and data else data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass

    # Attempt 3: complete truncated JSON then repair again
    completed = _complete_truncated_json(repaired)
    completed = _repair_json(completed)
    try:
        data = json.loads(completed)
        return data[0] if isinstance(data, list) and data else data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass

    return None


def _nuclear_extract(raw: str) -> dict:
    """Last resort: extract individual fields from raw text using regex.
    Returns a minimal dict with whatever we can find.
    """
    data: dict = {}

    def _grab(pattern: str, default: Any = "") -> Any:
        m = re.search(pattern, raw, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else default

    data["pr_url"]            = _grab(r'"pr_url"\s*:\s*"([^"]*)"')
    data["pr_title"]          = _grab(r'"pr_title"\s*:\s*"([^"]*)"')
    data["executive_summary"] = _grab(r'"executive_summary"\s*:\s*"([^"]{0,500})"')

    # Verdict
    v = _grab(r'"verdict"\s*:\s*"([^"]*)"')
    if "APPROVE" in v.upper() and "REQUEST" not in v.upper():
        data["verdict"] = "APPROVE"
    elif "REJECT" in v.upper():
        data["verdict"] = "REJECT"
    else:
        data["verdict"] = "REQUEST_CHANGES"

    # Scores
    scores: dict[str, Any] = {}
    for key in ("code_quality", "security", "performance", "overall"):
        val = _grab(rf'"{key}"\s*:\s*(\d+)')
        try:
            scores[key] = max(1, min(10, int(val)))
        except (ValueError, TypeError):
            scores[key] = 5
    data["scores"] = scores

    # Blocking issues — extract plain strings from the array
    bi_match = re.search(r'"blocking_issues"\s*:\s*\[(.*?)\]', raw, re.DOTALL)
    if bi_match:
        strings = re.findall(r'"([^"]{3,200})"', bi_match.group(1))
        data["blocking_issues"] = strings[:10]
    else:
        data["blocking_issues"] = []

    # Findings — try to extract minimal finding objects
    findings = []
    for fm in re.finditer(r'\{[^{}]*"title"\s*:\s*"([^"]+)"[^{}]*\}', raw):
        chunk = fm.group(0)
        title = re.search(r'"title"\s*:\s*"([^"]+)"', chunk)
        severity = re.search(r'"severity"\s*:\s*"([^"]+)"', chunk)
        category = re.search(r'"category"\s*:\s*"([^"]+)"', chunk)
        file_ = re.search(r'"file"\s*:\s*"([^"]+)"', chunk)
        desc = re.search(r'"description"\s*:\s*"([^"]+)"', chunk)
        if title:
            findings.append({
                "id": f"F{len(findings)+1:03d}",
                "title": title.group(1),
                "severity": severity.group(1) if severity else "info",
                "category": category.group(1) if category else "code_quality",
                "file": file_.group(1) if file_ else "",
                "description": desc.group(1) if desc else title.group(1),
                "fix": {"description": "", "code": ""},
                "references": [],
            })
    data["findings"] = findings[:20]

    return data


def _repair_json(s: str) -> str:
    """Repair common LLM JSON mistakes."""
    s = re.sub(r",\s*([}\]])", r"\1", s)       # trailing commas
    s = s.replace(": None", ": null").replace(":None", ":null")
    s = s.replace(": True", ": true").replace(":True", ":true")
    s = s.replace(": False", ": false").replace(":False", ":false")
    return s


def _complete_truncated_json(s: str) -> str:
    """Close a JSON string that was cut off mid-output.

    Handles three truncation cases:
    - Mid-string value:  ..."description":"incompl  → closes the string
    - Mid-key:           ...,"category":"v","incompl → removes dangling key, closes
    - Mid-comma:         ...,"category":"value",     → removes trailing comma, closes
    """
    in_string = False
    escape_next = False
    stack: list[str] = []

    for ch in s:
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in ('{', '['):
            stack.append(ch)
        elif ch == '}' and stack and stack[-1] == '{':
            stack.pop()
        elif ch == ']' and stack and stack[-1] == '[':
            stack.pop()

    # If truncated mid-string, close it
    result = (s + '"') if in_string else s

    # Remove trailing comma
    result = re.sub(r",\s*$", "", result.rstrip())

    # Remove dangling key with no value: ends with ,"some_key" before close
    # e.g. ...,"category":"v","dangling_key" → remove ,"dangling_key"
    result = re.sub(r',\s*"[^"\\]*"\s*$', "", result)

    # Remove trailing comma again (in case removing the dangling key left one)
    result = re.sub(r",\s*$", "", result.rstrip())

    # Close all open structures
    for opener in reversed(stack):
        result += '}' if opener == '{' else ']'

    return result


def _strip_problematic_fields(data: dict) -> None:
    """Normalise all fields in-place to safe types."""
    # blocking_issues: must be list of plain strings
    raw_bi = data.get("blocking_issues")
    if isinstance(raw_bi, list):
        data["blocking_issues"] = [
            item if isinstance(item, str) else
            (item.get("title") or item.get("description") or str(item))
            if isinstance(item, dict) else str(item)
            for item in raw_bi if item is not None
        ]
    else:
        data["blocking_issues"] = []

    # findings: normalise every entry
    if isinstance(data.get("findings"), list):
        cleaned = []
        for i, f in enumerate(data["findings"]):
            if not isinstance(f, dict):
                continue
            for key, default in [("id", f"F{i+1:03d}"), ("category", "code_quality"),
                                  ("severity", "info"), ("file", ""),
                                  ("title", ""), ("description", "")]:
                if not f.get(key):
                    f[key] = default
            fix = f.get("fix")
            if isinstance(fix, str):
                f["fix"] = {"description": fix, "code": ""}
            elif isinstance(fix, dict):
                fix["code"] = str(fix.get("code") or "")
                fix["description"] = str(fix.get("description") or "")
            else:
                f["fix"] = {"description": "", "code": ""}
            refs = f.get("references")
            f["references"] = [str(r) for r in refs if r is not None] if isinstance(refs, list) else []
            # Remove exploit_scenario if None to avoid confusion
            if f.get("exploit_scenario") is None:
                f.pop("exploit_scenario", None)
            cleaned.append(f)
        data["findings"] = cleaned

    # scores: fill any missing keys
    scores = data.get("scores")
    if not isinstance(scores, dict):
        data["scores"] = {"code_quality": 5, "security": 5, "performance": 5, "overall": 5}
    else:
        for key in ("code_quality", "security", "performance", "overall"):
            try:
                scores[key] = max(1, min(10, int(float(str(scores.get(key, 5))))))
            except (ValueError, TypeError):
                scores[key] = 5
        if not isinstance(scores.get("rationale"), dict):
            scores["rationale"] = {}

    # tool_usage_log: keep only dicts
    if isinstance(data.get("tool_usage_log"), list):
        data["tool_usage_log"] = [t for t in data["tool_usage_log"] if isinstance(t, dict)]
    else:
        data["tool_usage_log"] = []

    # verdict
    v = str(data.get("verdict", "")).upper().strip().replace("-", "_").replace(" ", "_")
    if "APPROVE" in v and "REQUEST" not in v:
        data["verdict"] = "APPROVE"
    elif "REJECT" in v:
        data["verdict"] = "REJECT"
    else:
        data["verdict"] = "REQUEST_CHANGES"

    # passes_completed
    try:
        data["passes_completed"] = max(1, int(data.get("passes_completed", 1)))
    except (ValueError, TypeError):
        data["passes_completed"] = 1


# ------------------------------------------------------------------ #
# Utility helpers                                                     #
# ------------------------------------------------------------------ #

_TOOL_PURPOSES = {
    "fetch_pr_metadata":    "Fetch PR title, author, and change statistics",
    "fetch_pr_diff":        "Fetch changed files and unified diffs",
    "fetch_file_content":   "Read full file content at PR head",
    "fetch_pr_commits":     "Fetch commit messages and authors",
    "analyze_complexity":   "Measure cyclomatic complexity and maintainability index",
    "analyze_syntax":       "Check for syntax errors",
    "count_code_metrics":   "Count LOC, SLOC, comment ratio",
    "run_bandit_scan":      "SAST security scan (OWASP / CWE)",
    "run_dependency_audit": "Scan dependencies for known CVEs",
    "extract_functions":    "Map function and class definitions",
    "search_patterns":      "Regex search for anti-patterns",
    "detect_redundant_code":"Detect unused imports, dead code, duplicate definitions",
    "detect_bugs":          "Detect common bugs: mutable defaults, bare excepts, etc.",
}


def _tool_purpose(name: str) -> str:
    return _TOOL_PURPOSES.get(name, name)


def _sanitize(inputs: dict) -> dict:
    out = {}
    for k, v in inputs.items():
        if isinstance(v, str) and len(v) > 300:
            out[k] = v[:300] + f"… [{len(v)} chars]"
        else:
            out[k] = v
    return out


def _result_summary(tool_name: str, result: dict) -> str:
    if not result.get("success"):
        return f"Error: {result.get('error', 'unknown')}"
    r = result.get("result", {})
    summaries = {
        "fetch_pr_metadata":    lambda: f"PR: \"{r.get('title', '')}\" by {r.get('author', '')} — {r.get('changed_files', 0)} files changed",
        "fetch_pr_diff":        lambda: f"{r.get('total_files', 0)} files in diff",
        "fetch_file_content":   lambda: f"{r.get('line_count', 0)} lines — {r.get('language', '')}",
        "fetch_pr_commits":     lambda: f"{r.get('count', 0)} commits",
        "analyze_complexity":   lambda: f"Avg complexity {r.get('average_complexity', 0)} — MI {r.get('maintainability_index', 0)} — {len(r.get('high_complexity_functions', []))} high-complexity fns",
        "analyze_syntax":       lambda: "Valid" if r.get("valid") else f"Syntax error: {r.get('errors', ['?'])[0]}",
        "count_code_metrics":   lambda: f"{r.get('sloc', 0)} SLOC, {r.get('comment_ratio_pct', 0)}% comments",
        "run_bandit_scan":      lambda: f"{r.get('total_issues', 0)} issues — HIGH:{r.get('metrics', {}).get('high', 0)} MED:{r.get('metrics', {}).get('medium', 0)}",
        "run_dependency_audit": lambda: f"{r.get('vulnerable_count', 0)} vulnerable packages out of {r.get('total_packages_scanned', 0)}",
        "extract_functions":    lambda: f"{r.get('count', 0)} definitions found",
        "search_patterns":      lambda: f"{r.get('total_matches', 0)} pattern matches across {r.get('patterns_searched', 0)} patterns",
        "detect_redundant_code":lambda: f"{r.get('issue_count', 0)} redundancy issues — categories: {', '.join(r.get('categories', [])) or 'none'}",
        "detect_bugs":          lambda: (
            f"{r.get('bug_count', 0)} bugs — "
            f"HIGH:{r.get('severity_breakdown', {}).get('high', 0)} "
            f"MED:{r.get('severity_breakdown', {}).get('medium', 0)} "
            f"LOW:{r.get('severity_breakdown', {}).get('low', 0)}"
        ),
    }
    fn = summaries.get(tool_name)
    try:
        return fn() if fn else "OK"
    except Exception:
        return "OK"
