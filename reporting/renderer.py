"""Rich terminal renderer for the final code review report."""

from __future__ import annotations

from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from agent.schemas import FinalReview, ReviewFinding, Severity, Category
from utils.logger import ReviewLogger


_SEVERITY_COLORS = {
    Severity.CRITICAL: "bold red",
    Severity.HIGH: "red",
    Severity.MEDIUM: "yellow",
    Severity.LOW: "cyan",
    Severity.INFO: "dim",
}

_SEVERITY_ICONS = {
    Severity.CRITICAL: "[CRITICAL]",
    Severity.HIGH: "[HIGH]",
    Severity.MEDIUM: "[MEDIUM]",
    Severity.LOW: "[LOW]",
    Severity.INFO: "[INFO]",
}

_CATEGORY_ICONS = {
    Category.BUG: "Bug",
    Category.SECURITY: "Security",
    Category.PERFORMANCE: "Performance",
    Category.CODE_QUALITY: "Quality",
}

_VERDICT_STYLES = {
    "APPROVE": ("green", "APPROVE"),
    "REQUEST_CHANGES": ("yellow", "REQUEST CHANGES"),
    "REJECT": ("bold red", "REJECT"),
}


def _score_color(score: int) -> str:
    if score >= 8:
        return "green"
    if score >= 6:
        return "yellow"
    return "red"


def _score_bar(score: int, width: int = 10) -> str:
    filled = round(score / 10 * width)
    return "█" * filled + "░" * (width - filled)


class ReportRenderer:
    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()

    def print_report(self, review: FinalReview, logger: ReviewLogger | None = None) -> None:
        self._print_header(review)
        self._print_scores(review)
        if review.findings:
            self._print_findings(review.findings)
        else:
            self.console.print("\n[green]No issues found.[/green]\n")
        if logger:
            self._print_tool_log(logger)
        self._print_verdict(review)

    # ------------------------------------------------------------------ #
    # Sections                                                             #
    # ------------------------------------------------------------------ #

    def _print_header(self, review: FinalReview) -> None:
        header = Text()
        header.append("Autonomous Code Review Report\n\n", style="bold white")
        header.append(f"PR:    {review.pr_title}\n", style="white")
        header.append(f"URL:   {review.pr_url}\n", style="dim")
        header.append(f"Passes completed: {review.passes_completed}\n", style="dim")
        self.console.print(Panel(header, title="[bold cyan]CODE REVIEW AGENT[/bold cyan]", border_style="cyan"))

    def _print_scores(self, review: FinalReview) -> None:
        table = Table(title="Scores", show_header=True, header_style="bold magenta", expand=False)
        table.add_column("Category", style="bold", min_width=16)
        table.add_column("Score", justify="center", min_width=7)
        table.add_column("Visual", min_width=12)

        rows = [
            ("Code Quality", review.scores.code_quality),
            ("Security", review.scores.security),
            ("Performance", review.scores.performance),
            ("Overall", review.scores.overall),
        ]

        for label, score in rows:
            color = _score_color(score)
            bar = _score_bar(score)
            table.add_row(label, f"[{color}]{score}/10[/{color}]", f"[{color}]{bar}[/{color}]")

        self.console.print(table)

    def _print_findings(self, findings: list[ReviewFinding]) -> None:
        # Sort by severity order
        order = {Severity.CRITICAL: 0, Severity.HIGH: 1, Severity.MEDIUM: 2,
                 Severity.LOW: 3, Severity.INFO: 4}
        sorted_findings = sorted(findings, key=lambda f: order.get(f.severity, 5))

        self.console.print(f"\n[bold]Findings ({len(findings)} total)[/bold]")

        for finding in sorted_findings:
            self._print_finding(finding)

    def _print_finding(self, finding: ReviewFinding) -> None:
        sev_color = _SEVERITY_COLORS[finding.severity]
        sev_icon = _SEVERITY_ICONS[finding.severity]
        cat_icon = _CATEGORY_ICONS.get(finding.category, str(finding.category))

        location = finding.file
        if finding.line_start:
            location += f":{finding.line_start}"
            if finding.line_end and finding.line_end != finding.line_start:
                location += f"-{finding.line_end}"

        title_text = Text()
        title_text.append(f"{sev_icon} ", style=sev_color)
        title_text.append(f"[{cat_icon}] ", style="bold")
        title_text.append(finding.title)

        body = Text()
        body.append(f"File: {location}\n", style="dim")
        body.append(f"ID:   {finding.id}\n\n", style="dim")
        body.append(finding.description)

        if finding.exploit_scenario:
            body.append("\n\nExploit scenario: ", style="bold red")
            body.append(finding.exploit_scenario)

        if finding.fix.description:
            body.append("\n\nFix: ", style="bold green")
            body.append(finding.fix.description)

        if finding.fix.code.strip():
            self.console.print(Panel(body, title=title_text, border_style=sev_color))
            lang = "python" if ".py" in finding.file else "text"
            self.console.print(
                Panel(
                    Syntax(finding.fix.code.strip(), lang, theme="monokai", word_wrap=True),
                    title="[green]Suggested Fix[/green]",
                    border_style="green",
                )
            )
        else:
            self.console.print(Panel(body, title=title_text, border_style=sev_color))

        if finding.references:
            self.console.print(f"  [dim]References: {', '.join(finding.references)}[/dim]")

    def _print_tool_log(self, logger: ReviewLogger) -> None:
        summary = logger.get_summary()
        table = Table(title="Tool Usage Log", show_header=True, header_style="bold yellow", expand=False)
        table.add_column("Tool", style="yellow", min_width=24)
        table.add_column("Calls", justify="right", min_width=6)

        for tool_name, count in sorted(summary["calls_by_tool"].items()):
            table.add_row(tool_name, str(count))

        table.add_row(
            "[bold]TOTAL[/bold]",
            f"[bold]{summary['total_calls']}[/bold]",
        )
        self.console.print(table)

    def _print_verdict(self, review: FinalReview) -> None:
        style, label = _VERDICT_STYLES.get(review.verdict, ("white", review.verdict))

        body = Text()
        body.append(f"{label}\n\n", style=f"bold {style}")
        body.append(review.executive_summary)

        if review.blocking_issues:
            body.append("\n\nBlocking issues: ", style="bold red")
            body.append(", ".join(review.blocking_issues))

        self.console.print(
            Panel(
                body,
                title="[bold]Final Verdict[/bold]",
                border_style=style,
                expand=False,
            )
        )
