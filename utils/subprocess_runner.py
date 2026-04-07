"""Safe subprocess wrapper for CLI security/analysis tools."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass


class SubprocessTimeoutError(Exception):
    pass


@dataclass
class SubprocessResult:
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool = False


def run_tool(
    command: list[str],
    input_data: str | None = None,
    timeout_seconds: int = 30,
    working_dir: str | None = None,
) -> SubprocessResult:
    """Run an external CLI tool safely.

    Never raises on non-zero exit codes (many tools like bandit return 1
    when they find issues). Only raises SubprocessTimeoutError on timeout.
    """
    try:
        result = subprocess.run(
            command,
            input=input_data,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=working_dir,
        )
        return SubprocessResult(
            stdout=result.stdout,
            stderr=result.stderr,
            returncode=result.returncode,
        )
    except subprocess.TimeoutExpired:
        raise SubprocessTimeoutError(
            f"Command {command[0]!r} timed out after {timeout_seconds}s"
        )
    except FileNotFoundError:
        return SubprocessResult(
            stdout="",
            stderr=f"Tool not found: {command[0]!r}. Is it installed?",
            returncode=127,
        )
