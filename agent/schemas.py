"""Pydantic models for the structured final review output.

Every field has a coercion validator so LLM hallucinations / type mismatches
never cause a hard validation failure.  The philosophy: parse what we can,
drop what we can't, never crash on schema errors.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ------------------------------------------------------------------ #
# Enums                                                               #
# ------------------------------------------------------------------ #

class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class Category(str, Enum):
    BUG = "bug"
    SECURITY = "security"
    PERFORMANCE = "performance"
    CODE_QUALITY = "code_quality"


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #

def _fuzzy_enum(value: Any, enum_cls: type[Enum], fallback: str) -> str:
    """Match value to an enum member, using substring matching as fallback."""
    if isinstance(value, enum_cls):
        return value.value
    if not isinstance(value, str):
        return fallback
    v = value.lower().strip().replace("-", "_").replace(" ", "_")
    # Exact match first
    for member in enum_cls:
        if member.value == v:
            return member.value
    # Substring match
    for member in enum_cls:
        if member.value in v or v in member.value:
            return member.value
    return fallback


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value)))
    except (ValueError, TypeError):
        return default


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


# ------------------------------------------------------------------ #
# Sub-models                                                          #
# ------------------------------------------------------------------ #

class FindingFix(BaseModel):
    model_config = ConfigDict(extra="ignore")

    description: str = ""
    code: str = ""

    @model_validator(mode="before")
    @classmethod
    def coerce(cls, v: Any) -> dict:
        if isinstance(v, str):
            return {"description": v, "code": ""}
        if isinstance(v, dict):
            return v
        return {"description": str(v) if v is not None else "", "code": ""}

    @field_validator("description", "code", mode="before")
    @classmethod
    def coerce_str(cls, v: Any) -> str:
        return str(v) if v is not None else ""


class ReviewFinding(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = "F000"
    category: Category = Category.CODE_QUALITY
    severity: Severity = Severity.INFO
    file: str = ""
    line_start: int | None = None
    line_end: int | None = None
    title: str = ""
    description: str = ""
    exploit_scenario: str | None = None
    fix: FindingFix = Field(default_factory=FindingFix)
    references: list[str] = Field(default_factory=list)

    @field_validator("category", mode="before")
    @classmethod
    def coerce_category(cls, v: Any) -> str:
        return _fuzzy_enum(v, Category, Category.CODE_QUALITY.value)

    @field_validator("severity", mode="before")
    @classmethod
    def coerce_severity(cls, v: Any) -> str:
        return _fuzzy_enum(v, Severity, Severity.INFO.value)

    @field_validator("line_start", "line_end", mode="before")
    @classmethod
    def coerce_line(cls, v: Any) -> int | None:
        if v is None:
            return None
        try:
            return int(float(str(v)))
        except (ValueError, TypeError):
            return None

    @field_validator("fix", mode="before")
    @classmethod
    def coerce_fix(cls, v: Any) -> Any:
        if isinstance(v, str):
            return {"description": v, "code": ""}
        if v is None:
            return {"description": "", "code": ""}
        return v

    @field_validator("references", mode="before")
    @classmethod
    def coerce_references(cls, v: Any) -> list[str]:
        if not isinstance(v, list):
            return []
        return [str(item) for item in v if item is not None]

    @field_validator("exploit_scenario", mode="before")
    @classmethod
    def coerce_exploit(cls, v: Any) -> str | None:
        if v is None or v == "":
            return None
        return str(v)

    @field_validator("id", "file", "title", "description", mode="before")
    @classmethod
    def coerce_str(cls, v: Any) -> str:
        if v is None:
            return ""
        return str(v)


class ToolUsageEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")

    tool: str = ""
    purpose: str = ""
    key_finding: str = ""

    @model_validator(mode="before")
    @classmethod
    def coerce(cls, v: Any) -> dict:
        if not isinstance(v, dict):
            return {"tool": str(v), "purpose": "", "key_finding": ""}
        return v

    @field_validator("tool", "purpose", "key_finding", mode="before")
    @classmethod
    def coerce_str(cls, v: Any) -> str:
        return str(v) if v is not None else ""


class ReviewScores(BaseModel):
    model_config = ConfigDict(extra="ignore")

    code_quality: int = 5
    security: int = 5
    performance: int = 5
    overall: int = 5
    rationale: dict[str, str] = Field(default_factory=dict)

    @field_validator("code_quality", "security", "performance", "overall", mode="before")
    @classmethod
    def coerce_score(cls, v: Any) -> int:
        return _clamp(_to_int(v, 5), 1, 10)

    @field_validator("rationale", mode="before")
    @classmethod
    def coerce_rationale(cls, v: Any) -> dict[str, str]:
        if not isinstance(v, dict):
            return {}
        return {str(k): str(val) for k, val in v.items() if val is not None}


# ------------------------------------------------------------------ #
# Top-level model                                                     #
# ------------------------------------------------------------------ #

class FinalReview(BaseModel):
    model_config = ConfigDict(extra="ignore")

    pr_url: str = ""
    pr_title: str = ""
    executive_summary: str = ""
    passes_completed: int = 1
    tool_usage_log: list[ToolUsageEntry] = Field(default_factory=list)
    findings: list[ReviewFinding] = Field(default_factory=list)
    scores: ReviewScores = Field(default_factory=ReviewScores)
    verdict: Literal["APPROVE", "REQUEST_CHANGES", "REJECT"] = "REQUEST_CHANGES"
    blocking_issues: list[str] = Field(default_factory=list)

    @field_validator("pr_url", "pr_title", "executive_summary", mode="before")
    @classmethod
    def coerce_str(cls, v: Any) -> str:
        return str(v) if v is not None else ""

    @field_validator("passes_completed", mode="before")
    @classmethod
    def coerce_passes(cls, v: Any) -> int:
        return max(1, _to_int(v, 1))

    @field_validator("verdict", mode="before")
    @classmethod
    def coerce_verdict(cls, v: Any) -> str:
        if not isinstance(v, str):
            return "REQUEST_CHANGES"
        v_up = v.upper().strip().replace(" ", "_").replace("-", "_")
        if v_up in {"APPROVE", "APPROVED", "LGTM"}:
            return "APPROVE"
        if v_up in {"REJECT", "REJECTED", "BLOCK"}:
            return "REJECT"
        if "APPROVE" in v_up and "REQUEST" not in v_up:
            return "APPROVE"
        if "REJECT" in v_up:
            return "REJECT"
        return "REQUEST_CHANGES"

    @field_validator("blocking_issues", mode="before")
    @classmethod
    def coerce_blocking_issues(cls, v: Any) -> list[str]:
        """LLM sometimes outputs full finding objects instead of plain strings."""
        if not isinstance(v, list):
            return []
        result: list[str] = []
        for item in v:
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, dict):
                title    = item.get("title") or item.get("description") or item.get("id") or ""
                severity = str(item.get("severity", "")).upper()
                file_    = item.get("file", "")
                parts    = [p for p in [severity, title, f"in {file_}" if file_ else ""] if p]
                result.append(" — ".join(parts) if parts else str(item))
            elif item is not None:
                result.append(str(item))
        return result

    @field_validator("findings", mode="before")
    @classmethod
    def coerce_findings(cls, v: Any) -> list[dict]:
        """Drop anything that isn't a dict; the sub-model handles the rest."""
        if not isinstance(v, list):
            return []
        return [item for item in v if isinstance(item, dict)]

    @field_validator("tool_usage_log", mode="before")
    @classmethod
    def coerce_tool_log(cls, v: Any) -> list:
        if not isinstance(v, list):
            return []
        return [item for item in v if item is not None]

    @field_validator("scores", mode="before")
    @classmethod
    def coerce_scores(cls, v: Any) -> Any:
        if v is None:
            return {}
        return v

