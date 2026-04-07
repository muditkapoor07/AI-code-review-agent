"""Tool call logger with optional Rich verbose output."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax


@dataclass
class ToolCallRecord:
    timestamp: datetime
    pass_number: int
    tool_name: str
    inputs: dict
    result_summary: str
    success: bool
    duration_ms: float


class ReviewLogger:
    def __init__(self, verbose: bool, console: Console) -> None:
        self.verbose = verbose
        self.console = console
        self.call_log: list[ToolCallRecord] = []
        self._current_pass = 0

    def log_pass_start(self, pass_number: int) -> None:
        self._current_pass = pass_number
        if self.verbose:
            self.console.print(
                f"\n[bold cyan]── Pass {pass_number} ──────────────────────────[/bold cyan]"
            )

    def log_pass_end(self, pass_number: int, stop_reason: str) -> None:
        if self.verbose:
            self.console.print(
                f"[dim]Pass {pass_number} ended — stop_reason: {stop_reason}[/dim]"
            )

    def log_tool_call(
        self,
        tool_name: str,
        inputs: dict,
        result: dict,
        duration_ms: float = 0.0,
    ) -> None:
        success = result.get("success", True)
        result_summary = json.dumps(result, default=str)[:300]

        self.call_log.append(
            ToolCallRecord(
                timestamp=datetime.utcnow(),
                pass_number=self._current_pass,
                tool_name=tool_name,
                inputs=inputs,
                result_summary=result_summary,
                success=success,
                duration_ms=duration_ms,
            )
        )

        if self.verbose:
            status = "[green]OK[/green]" if success else "[red]ERROR[/red]"
            self.console.print(
                Panel(
                    Syntax(
                        json.dumps({"inputs": inputs, "result_preview": result_summary}, indent=2),
                        "json",
                        theme="monokai",
                        word_wrap=True,
                    ),
                    title=f"[bold yellow]TOOL:[/bold yellow] {tool_name}  {status}  "
                          f"[dim]{duration_ms:.0f}ms[/dim]",
                    border_style="yellow",
                    expand=False,
                )
            )

    def get_summary(self) -> dict:
        calls_by_tool: dict[str, int] = {}
        total_ms = 0.0
        for record in self.call_log:
            calls_by_tool[record.tool_name] = calls_by_tool.get(record.tool_name, 0) + 1
            total_ms += record.duration_ms
        return {
            "total_calls": len(self.call_log),
            "calls_by_tool": calls_by_tool,
            "total_duration_ms": total_ms,
            "passes": self._current_pass,
        }
