"""Pydantic models for the structured final review output."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


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


class FindingFix(BaseModel):
    description: str
    code: str = ""


class ReviewFinding(BaseModel):
    id: str = Field(description="e.g. F001")
    category: Category
    severity: Severity
    file: str
    line_start: int | None = None
    line_end: int | None = None
    title: str
    description: str
    exploit_scenario: str | None = None
    fix: FindingFix
    references: list[str] = Field(default_factory=list)


class ToolUsageEntry(BaseModel):
    tool: str
    purpose: str
    key_finding: str


class ReviewScores(BaseModel):
    code_quality: int = Field(ge=1, le=10)
    security: int = Field(ge=1, le=10)
    performance: int = Field(ge=1, le=10)
    overall: int = Field(ge=1, le=10)
    rationale: dict[str, str] = Field(default_factory=dict)


class FinalReview(BaseModel):
    pr_url: str
    pr_title: str
    executive_summary: str
    passes_completed: int
    tool_usage_log: list[ToolUsageEntry] = Field(default_factory=list)
    findings: list[ReviewFinding] = Field(default_factory=list)
    scores: ReviewScores
    verdict: Literal["APPROVE", "REQUEST_CHANGES", "REJECT"]
    blocking_issues: list[str] = Field(default_factory=list)

    @property
    def critical_findings(self) -> list[ReviewFinding]:
        return [f for f in self.findings if f.severity == Severity.CRITICAL]

    @property
    def high_findings(self) -> list[ReviewFinding]:
        return [f for f in self.findings if f.severity == Severity.HIGH]

    @property
    def security_findings(self) -> list[ReviewFinding]:
        return [f for f in self.findings if f.category == Category.SECURITY]
