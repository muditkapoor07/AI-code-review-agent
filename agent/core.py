"""Core agent loop using Groq (OpenAI-compatible API)."""

from __future__ import annotations

import json
import re
import time
from typing import Any, Callable

from openai import OpenAI

from agent.prompts import SYSTEM_PROMPT, INITIAL_USER_MESSAGE, SNIPPET_USER_MESSAGE
from agent.schemas import FinalReview
from github.client import GitHubClient
from tools.registry import ToolRegistry
from utils.logger import ReviewLogger


_DEFAULT_MODEL = "llama-3.3-70b-versatile"
_DEFAULT_MAX_PASSES = 10
_MAX_TOKENS = 2048
_MAX_TOOL_RESULT_CHARS = 1500  # truncate large tool results in history
_MAX_HISTORY_MESSAGES = 12     # keep only recent messages to stay under TPM
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
        raw_output = self._agentic_loop(messages)
        return self._parse_final_review(raw_output, pr_url)

    def run_snippet(self, code: str, language: str, filename: str) -> FinalReview:
        self._emit({"type": "status", "message": f"Reviewing {filename} ({language})"})

        initial_message = SNIPPET_USER_MESSAGE.format(
            filename=filename, language=language, code=code,
        )
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": initial_message},
        ]
        raw_output = self._agentic_loop(messages, source_ref=f"snippet:{filename}")
        return self._parse_final_review(raw_output, f"snippet:{filename}")

    # ------------------------------------------------------------------ #
    # Agent loop                                                           #
    # ------------------------------------------------------------------ #

    def _agentic_loop(self, messages: list[dict], source_ref: str = "") -> str:
        pass_number = 0
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
                    return self._force_final_answer(messages)
                if "429" in err_str or "rate_limit_exceeded" in err_str:
                    import re as _re
                    wait = _re.search(r'try again in ([^.]+)', err_str)
                    wait_msg = f"Rate limit hit — try again in {wait.group(1)}" if wait else "Rate limit hit — please wait before retrying"
                    self._emit({"type": "error", "message": wait_msg})
                    raise
                raise
            choice = response.choices[0]
            msg = choice.message
            stop_reason = choice.finish_reason  # "stop" | "tool_calls" | "length"

            self._logger.log_pass_end(pass_number, stop_reason)
            self._emit({"type": "pass_end", "pass": pass_number, "stop_reason": stop_reason})

            # Emit any text content
            if msg.content and msg.content.strip():
                self._emit({"type": "thinking", "text": msg.content[:500]})

            if stop_reason == "stop" or not msg.tool_calls:
                text = msg.content or ""
                # If the model stopped but didn't produce the final review, push it
                if "<final_review>" not in text:
                    messages.append({"role": "assistant", "content": text})
                    return self._force_final_answer(messages)
                return text

            if stop_reason == "tool_calls" or msg.tool_calls:
                # Append assistant message — strip fields Groq doesn't support
                msg_dict = msg.model_dump(exclude_unset=True, exclude_none=True)
                msg_dict.pop("annotations", None)
                msg_dict.pop("audio", None)
                msg_dict.pop("refusal", None)
                messages.append(msg_dict)

                # Execute each tool call
                for tc in msg.tool_calls:
                    tool_name = tc.function.name
                    try:
                        tool_input = json.loads(tc.function.arguments)
                    except (json.JSONDecodeError, Exception):
                        tool_input = {}
                        self._emit({"type": "tool_end", "tool": tool_name, "success": False,
                                    "summary": "Skipped — malformed arguments", "duration_ms": 0})
                        messages.append({"role": "tool", "tool_call_id": tc.id,
                                         "content": '{"success":false,"error":"malformed arguments — skipped","result":{}}'})
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

                    # Append tool result — truncate to stay within token limits
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

        return self._force_final_answer(messages)

    def _call_groq(self, messages: list[dict], tools: list[dict]):
        # Keep system message + last N messages to stay under TPM limits
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
                "STOP. Do not call any more tools. Output the final review JSON NOW.\n"
                "Wrap it in <final_review>...</final_review> tags. Output NOTHING else.\n\n"
                "STRICT TYPE RULES — violating these will break the parser:\n"
                "• blocking_issues: LIST OF PLAIN STRINGS only — e.g. [\"SQL injection in login.py\"]. "
                "NEVER put objects/dicts in this list.\n"
                "• findings[].category: must be exactly one of: bug, security, performance, code_quality\n"
                "• findings[].severity: must be exactly one of: critical, high, medium, low, info\n"
                "• findings[].fix: object with 'description' (string) and 'code' (string) keys\n"
                "• scores.rationale: object where each value is a plain string sentence\n"
                "• verdict: exactly APPROVE, REQUEST_CHANGES, or REJECT\n"
                "• passes_completed: integer\n\n"
                "Required fields: pr_url, pr_title, executive_summary, passes_completed, "
                "tool_usage_log (list), findings (list), scores (code_quality/security/"
                "performance/overall each 1-10, plus rationale object), verdict, blocking_issues."
            ),
        })
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=_MAX_TOKENS,
            messages=messages,
        )
        return response.choices[0].message.content or ""

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_final_review(raw_output: str, source_ref: str) -> FinalReview:
        # ── 1. Extract JSON string ──────────────────────────────────────
        match = re.search(r"<final_review>(.*?)</final_review>", raw_output, re.DOTALL)
        if not match:
            json_match = re.search(r"\{[\s\S]*\"verdict\"[\s\S]*\}", raw_output)
            if json_match:
                json_str = json_match.group(0)
            else:
                raise AgentLoopError(
                    "Agent did not produce a <final_review> block.\n\n"
                    f"Raw output:\n{raw_output[:2000]}"
                )
        else:
            json_str = match.group(1).strip()

        # Strip markdown code fences
        json_str = re.sub(r"^```(?:json)?\s*", "", json_str.strip())
        json_str = re.sub(r"\s*```$", "", json_str)

        # Pre-unwrap list wrapper at string level before parsing
        # Model sometimes outputs [{...}] — strip the outer [ ] so json.loads gives a dict
        stripped = json_str.strip()
        if stripped.startswith("["):
            # Find the first { and the matching last } to extract the inner object
            inner_match = re.search(r"\{[\s\S]*\}", stripped)
            if inner_match:
                json_str = inner_match.group(0)

        # ── 2. Parse JSON ───────────────────────────────────────────────
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            # Try repairing common LLM mistakes: trailing commas, Python booleans
            repaired = _repair_json(json_str)
            try:
                data = json.loads(repaired)
            except json.JSONDecodeError as exc:
                raise AgentLoopError(
                    f"Failed to parse final_review JSON: {exc}\n\n{json_str[:1000]}"
                )

        # Post-parse safety: unwrap any remaining list wrapping
        if isinstance(data, list):
            data = next((item for item in data if isinstance(item, dict)), {})

        if not isinstance(data, dict):
            raise AgentLoopError(f"Expected JSON object, got {type(data).__name__}: {json_str[:500]}")

        data.setdefault("pr_url", source_ref)

        # ── 3. Validate with Pydantic (validators handle coercion) ──────
        try:
            return FinalReview.model_validate(data)
        except Exception as exc:
            # ── 4. Last-resort: strip complex nested fields and retry ───
            _strip_problematic_fields(data)
            try:
                return FinalReview.model_validate(data)
            except Exception as exc2:
                raise AgentLoopError(
                    f"Final review schema validation failed after recovery attempt: {exc2}\n"
                    f"Original error: {exc}"
                )


# ------------------------------------------------------------------ #
# Internal helpers                                                     #
# ------------------------------------------------------------------ #

def _repair_json(s: str) -> str:
    """Best-effort repair of common LLM JSON mistakes."""
    # Remove trailing commas before } or ]
    s = re.sub(r",\s*([}\]])", r"\1", s)
    # Replace Python-style None/True/False
    s = s.replace(": None", ": null").replace(":None", ":null")
    s = s.replace(": True", ": true").replace(":True", ":true")
    s = s.replace(": False", ": false").replace(":False", ":false")
    return s


def _strip_problematic_fields(data: dict) -> None:
    """Mutate data in-place: replace fields that commonly fail with safe defaults.

    Called only when the first validation attempt fails — a last-resort path.
    """
    # blocking_issues: ensure list of strings
    if "blocking_issues" in data:
        raw = data["blocking_issues"]
        if isinstance(raw, list):
            data["blocking_issues"] = [
                item if isinstance(item, str) else
                (item.get("title") or item.get("description") or str(item))
                if isinstance(item, dict) else str(item)
                for item in raw
            ]
        else:
            data["blocking_issues"] = []

    # findings: drop any entry missing required fields
    if "findings" in data and isinstance(data["findings"], list):
        cleaned = []
        for f in data["findings"]:
            if not isinstance(f, dict):
                continue
            # Use 'or' assignment — setdefault won't overwrite None values
            for key, default in [("id", "F000"), ("category", "code_quality"),
                                  ("severity", "info"), ("file", ""),
                                  ("title", ""), ("description", "")]:
                if not f.get(key):
                    f[key] = default
            # fix: coerce string/None → dict with string fields
            fix = f.get("fix")
            if isinstance(fix, str):
                f["fix"] = {"description": fix, "code": ""}
            elif isinstance(fix, dict):
                fix["code"] = fix.get("code") or ""
                fix["description"] = fix.get("description") or ""
            else:
                f["fix"] = {"description": "", "code": ""}
            # references must be list of strings
            refs = f.get("references")
            if not isinstance(refs, list):
                f["references"] = []
            else:
                f["references"] = [str(r) for r in refs if r is not None]
            cleaned.append(f)
        data["findings"] = cleaned

    # scores: fill missing keys with defaults
    scores = data.get("scores")
    if not isinstance(scores, dict):
        data["scores"] = {"code_quality": 5, "security": 5, "performance": 5, "overall": 5}
    else:
        for key in ("code_quality", "security", "performance", "overall"):
            if key not in scores:
                scores[key] = 5

    # tool_usage_log: keep only dicts
    if "tool_usage_log" in data and isinstance(data["tool_usage_log"], list):
        data["tool_usage_log"] = [
            t for t in data["tool_usage_log"] if isinstance(t, dict)
        ]

    # verdict: normalise
    v = data.get("verdict", "")
    if not isinstance(v, str) or v.upper() not in {"APPROVE", "REQUEST_CHANGES", "REJECT"}:
        data["verdict"] = "REQUEST_CHANGES"

    # passes_completed: int
    try:
        data["passes_completed"] = int(data.get("passes_completed", 1))
    except (ValueError, TypeError):
        data["passes_completed"] = 1


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
        "fetch_pr_metadata": lambda: f"PR: \"{r.get('title', '')}\" by {r.get('author', '')} — {r.get('changed_files', 0)} files changed",
        "fetch_pr_diff":     lambda: f"{r.get('total_files', 0)} files in diff",
        "fetch_file_content":lambda: f"{r.get('line_count', 0)} lines — {r.get('language', '')}",
        "fetch_pr_commits":  lambda: f"{r.get('count', 0)} commits",
        "analyze_complexity":lambda: f"Avg complexity {r.get('average_complexity', 0)} — MI {r.get('maintainability_index', 0)} — {len(r.get('high_complexity_functions', []))} high-complexity fns",
        "analyze_syntax":    lambda: "Valid" if r.get("valid") else f"Syntax error: {r.get('errors', ['?'])[0]}",
        "count_code_metrics":lambda: f"{r.get('sloc', 0)} SLOC, {r.get('comment_ratio_pct', 0)}% comments",
        "run_bandit_scan":   lambda: f"{r.get('total_issues', 0)} issues — HIGH:{r.get('metrics', {}).get('high', 0)} MED:{r.get('metrics', {}).get('medium', 0)}",
        "run_dependency_audit": lambda: f"{r.get('vulnerable_count', 0)} vulnerable packages out of {r.get('total_packages_scanned', 0)}",
        "extract_functions": lambda: f"{r.get('count', 0)} definitions found",
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
