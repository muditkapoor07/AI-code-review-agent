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

# Code snippet preview limits — large code is shown truncated in the user
# message (to stay within TPM), but tools always receive the full code
# via ToolRegistry.set_snippet() regardless of size.
_MAX_SNIPPET_PREVIEW_LINES = 120   # lines shown to LLM in the user message
_MAX_SNIPPET_PREVIEW_CHARS = 4000  # char cap on the preview


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
        total_lines = code.count("\n") + 1
        total_chars = len(code)
        self._emit({
            "type": "status",
            "message": f"Reviewing {filename} ({language}) — {total_lines} lines, {total_chars:,} chars",
        })

        # Register the FULL code with the tool registry — tools will always
        # receive the complete source regardless of what preview the LLM sees.
        self._registry.set_snippet(code, filename)

        # Build a preview for the LLM context (keeps user message within TPM limits)
        lines = code.splitlines()
        if total_lines > _MAX_SNIPPET_PREVIEW_LINES or total_chars > _MAX_SNIPPET_PREVIEW_CHARS:
            preview = "\n".join(lines[:_MAX_SNIPPET_PREVIEW_LINES])
            if len(preview) > _MAX_SNIPPET_PREVIEW_CHARS:
                preview = preview[:_MAX_SNIPPET_PREVIEW_CHARS]
            hidden = total_lines - _MAX_SNIPPET_PREVIEW_LINES
            display_code = (
                preview
                + f"\n# ... [{hidden} more lines hidden from context — "
                "full code is automatically passed to all analysis tools]"
            )
        else:
            display_code = code

        initial_message = SNIPPET_USER_MESSAGE.format(
            filename=filename, language=language, code=display_code,
            total_lines=total_lines,
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
                "Output ONLY this JSON inside <final_review>...</final_review> tags. "
                "No other text. Do NOT include tool_usage_log. Keep findings to max 5. "
                "category MUST be one of: bug|security|performance|code_quality (underscore). "
                "severity MUST be one of: critical|high|medium|low|info. "
                "blocking_issues MUST always be [] (empty array).\n\n"
                '<final_review>{"pr_url":"","pr_title":"","executive_summary":"1-2 sentences",'
                '"findings":[{"id":"F001","category":"code_quality","severity":"info","file":"",'
                '"title":"","description":"","fix":{"description":"","code":""},"references":[]}],'
                '"scores":{"code_quality":5,"security":5,"performance":5,"overall":5,'
                '"rationale":{"code_quality":"","security":"","performance":"","overall":""}},'
                '"verdict":"APPROVE","blocking_issues":[]}</final_review>'
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

        # ── Step 0: strip tool_usage_log to cut JSON size before parsing ─
        stripped_raw = _strip_tool_usage_log(raw_output)

        # ── Step 1: extract the JSON string ────────────────────────────
        json_str = _extract_json_str(stripped_raw)

        # ── Step 2: parse with progressive recovery ─────────────────────
        data = _parse_with_recovery(json_str)

        # ── Step 3: if we still have nothing, nuclear regex extraction ──
        if data is None:
            data = _nuclear_extract(raw_output)  # use original for more context

        # ── Step 4: fill mandatory fields from our own tracking ─────────
        # Coerce to dict — handle list-wrapped dicts and any other type
        if isinstance(data, list):
            data = next((item for item in data if isinstance(item, dict)), {})
        if not isinstance(data, dict):
            data = {}

        if not data.get("pr_url"):      data["pr_url"] = source_ref
        if not data.get("pr_title"):    data["pr_title"] = ""
        if not data.get("executive_summary"): data["executive_summary"] = "Review completed."
        data["passes_completed"] = pass_count
        data["tool_usage_log"] = tool_log  # always inject ours, not LLM's
        if "findings" not in data:      data["findings"] = []
        if "scores" not in data:        data["scores"] = {}
        if not data.get("verdict"):     data["verdict"] = "REQUEST_CHANGES"
        # ALWAYS discard LLM-produced blocking_issues — it's the #1 crash source
        # We recompute it from validated findings after model_validate
        data["blocking_issues"] = []

        # ── Step 5: normalise and coerce ────────────────────────────────
        try:
            _strip_problematic_fields(data)
        except Exception:
            # Normalisation failed — reset to safe minimal state and continue
            data["findings"] = []
            data["scores"] = {"code_quality": 5, "security": 5,
                              "performance": 5, "overall": 5, "rationale": {}}
            data["blocking_issues"] = []

        # ── Step 6: Pydantic validate, then compute blocking_issues ourselves ─
        for attempt in range(2):
            try:
                review = FinalReview.model_validate(data)
                # Derive blocking_issues from validated finding objects (guaranteed correct types)
                review.blocking_issues = [
                    f"{f.severity.value.upper()} — {f.title}"
                    + (f" in {f.file}" if f.file else "")
                    for f in review.findings
                    if f.severity.value in ("critical", "high")
                ]
                return review
            except Exception:
                if attempt == 0:
                    # Strip findings on second attempt — they may be malformed
                    data["findings"] = []
                    data["scores"] = {"code_quality": 5, "security": 5,
                                      "performance": 5, "overall": 5, "rationale": {}}
                else:
                    raise AgentLoopError("Could not build a valid review even with stripped findings")


# ------------------------------------------------------------------ #
# JSON extraction & repair helpers                                    #
# ------------------------------------------------------------------ #

def _extract_first_object(s: str) -> str | None:
    """Extract the first complete JSON object from s using bracket counting."""
    depth = 0
    in_str = False
    esc = False
    start = None
    for i, ch in enumerate(s):
        if esc:
            esc = False
            continue
        if ch == "\\" and in_str:
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                return s[start : i + 1]
    return None


def _extract_json_str(raw: str) -> str:
    """Extract the JSON string from raw LLM output."""
    # Try <final_review> tags first
    match = re.search(r"<final_review>(.*?)</final_review>", raw, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
    else:
        # Fallback: find {…} block containing "verdict"; if not found, take any {…}
        json_match = (
            re.search(r"\{[\s\S]*\"verdict\"[\s\S]*\}", raw)
            or re.search(r"\{[\s\S]+\}", raw)
        )
        json_str = json_match.group(0) if json_match else raw.strip()

    # Strip markdown fences
    json_str = re.sub(r"^```(?:json)?\s*", "", json_str.strip())
    json_str = re.sub(r"\s*```$", "", json_str)

    # Unwrap list wrapper [{…}] → first {…} using bracket counting (not greedy regex)
    stripped = json_str.strip()
    if stripped.startswith("["):
        extracted = _extract_first_object(stripped)
        if extracted:
            json_str = extracted

    return json_str


def _coerce_to_dict(data: Any) -> dict | None:
    """Safely coerce parsed JSON data to a dict, handling list wrappers."""
    if isinstance(data, dict):
        return data
    if isinstance(data, list):
        return next((item for item in data if isinstance(item, dict)), None)
    return None


def _parse_with_recovery(json_str: str) -> dict | None:
    """Try to parse JSON with up to 3 recovery attempts. Returns dict or None."""
    # Attempt 1: direct parse
    try:
        return _coerce_to_dict(json.loads(json_str))
    except (json.JSONDecodeError, Exception):
        pass

    # Attempt 2: repair common mistakes
    repaired = _repair_json(json_str)
    try:
        return _coerce_to_dict(json.loads(repaired))
    except (json.JSONDecodeError, Exception):
        pass

    # Attempt 3: complete truncated JSON then repair again
    completed = _complete_truncated_json(repaired)
    completed = _repair_json(completed)
    try:
        return _coerce_to_dict(json.loads(completed))
    except (json.JSONDecodeError, Exception):
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


def _strip_tool_usage_log(raw: str) -> str:
    """Remove tool_usage_log array from raw output using bracket counting.
    This shrinks the JSON before parsing so truncation is less likely.
    """
    key = '"tool_usage_log"'
    idx = raw.find(key)
    if idx == -1:
        return raw
    colon = raw.find(":", idx + len(key))
    if colon == -1:
        return raw
    start = colon + 1
    while start < len(raw) and raw[start] in " \t\n\r":
        start += 1
    if start >= len(raw) or raw[start] != "[":
        return raw
    # Bracket-count to find the closing ]
    depth = 0
    in_str = False
    esc = False
    end = -1
    for i in range(start, len(raw)):
        ch = raw[i]
        if esc:
            esc = False
            continue
        if ch == "\\" and in_str:
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end == -1:
        return raw  # truncated inside tool_usage_log — leave as-is
    # Splice out: ,?"tool_usage_log":[...]
    pre = raw[:idx].rstrip()
    trailing_comma = pre.endswith(",")
    if trailing_comma:
        pre = pre[:-1].rstrip()
    post = raw[end + 1:].lstrip()
    if post.startswith(","):
        post = post[1:].lstrip()
    separator = "," if (pre and not pre.endswith("{") and post and not post.startswith("}")) else ""
    return pre + separator + post


_VALID_CATEGORIES = {"bug", "security", "performance", "code_quality"}
_VALID_SEVERITIES = {"critical", "high", "medium", "low", "info"}


def _normalize_enum(value: Any, valid: set[str], fallback: str) -> str:
    """Normalize a string to a valid enum value with fuzzy matching."""
    s = str(value).lower().strip().replace(" ", "_").replace("-", "_")
    if s in valid:
        return s
    for v in valid:
        if v in s or s in v:
            return v
    return fallback


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
            for key, default in [("id", f"F{i+1:03d}"), ("file", ""),
                                  ("title", ""), ("description", "")]:
                if not f.get(key):
                    f[key] = default
            # Always normalize enum fields — Pydantic coercion is backup only
            f["category"] = _normalize_enum(f.get("category", ""), _VALID_CATEGORIES, "code_quality")
            f["severity"]  = _normalize_enum(f.get("severity", ""),  _VALID_SEVERITIES,  "info")
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
