"""Autonomous Code Review Agent — entry point."""

from __future__ import annotations

import argparse
import json
import os
import sys

from dotenv import load_dotenv
from rich.console import Console

load_dotenv()


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        print(f"Error: {name} environment variable is not set.", file=sys.stderr)
        print(f"Copy .env.example to .env and fill in your keys.", file=sys.stderr)
        sys.exit(1)
    return value


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="code-review",
        description="Autonomous AI code review agent powered by Claude",
    )
    parser.add_argument(
        "pr_url",
        help="GitHub PR URL, e.g. https://github.com/owner/repo/pull/123",
    )
    parser.add_argument(
        "--max-passes",
        type=int,
        default=10,
        metavar="N",
        help="Maximum agent investigation passes (default: 10)",
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-6",
        help="Claude model to use (default: claude-sonnet-4-6)",
    )
    parser.add_argument(
        "--output-json",
        metavar="FILE",
        help="Write the final review JSON to this file",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print each tool call as it happens",
    )
    return parser


def main_cli() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    anthropic_key = _require_env("ANTHROPIC_API_KEY")
    github_token = _require_env("GITHUB_TOKEN")

    # Late imports so --help works without all dependencies installed
    import anthropic
    from github.client import GitHubClient
    from tools.registry import ToolRegistry
    from agent.core import CodeReviewAgent, AgentLoopError
    from reporting.renderer import ReportRenderer
    from utils.logger import ReviewLogger

    console = Console()
    logger = ReviewLogger(verbose=args.verbose, console=console)
    renderer = ReportRenderer(console=console)

    console.print(f"\n[bold cyan]Starting code review for:[/bold cyan] {args.pr_url}\n")

    with GitHubClient(token=github_token) as gh_client:
        registry = ToolRegistry(github_client=gh_client)
        anthropic_client = anthropic.Anthropic(api_key=anthropic_key)

        agent = CodeReviewAgent(
            anthropic_client=anthropic_client,
            github_client=gh_client,
            tool_registry=registry,
            logger=logger,
            model=args.model,
            max_passes=args.max_passes,
        )

        try:
            review = agent.run(args.pr_url)
        except AgentLoopError as exc:
            console.print(f"\n[bold red]Agent loop error:[/bold red] {exc}")
            sys.exit(1)
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted by user.[/yellow]")
            sys.exit(130)

    renderer.print_report(review, logger=logger)

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(review.model_dump(), f, indent=2, default=str)
        console.print(f"\n[dim]Review saved to: {args.output_json}[/dim]")


if __name__ == "__main__":
    main_cli()
