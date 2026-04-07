"""Security scanning tools: Bandit SAST + pip-audit CVE checking."""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any

from utils.subprocess_runner import run_tool, SubprocessTimeoutError


def run_bandit_scan(source_code: str, filename: str) -> dict[str, Any]:
    """Run Bandit static security analysis on Python source code."""
    suffix = os.path.splitext(filename)[1] or ".py"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=suffix,
            delete=False,
            encoding="utf-8",
        ) as tmp:
            tmp.write(source_code)
            tmp_path = tmp.name

        result = run_tool(
            ["bandit", "-f", "json", "-q", tmp_path],
            timeout_seconds=30,
        )

        if result.returncode == 127:
            return {
                "success": False,
                "error": "bandit is not installed. Run: pip install bandit",
                "result": {},
            }

        # bandit exits 1 when issues found, 0 when clean — both are valid
        if not result.stdout.strip():
            return {
                "success": True,
                "result": {
                    "filename": filename,
                    "issues": [],
                    "metrics": {"high": 0, "medium": 0, "low": 0},
                    "note": result.stderr[:500] if result.stderr else "No output",
                },
            }

        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError:
            return {
                "success": False,
                "error": f"Failed to parse bandit JSON output: {result.stdout[:300]}",
                "result": {},
            }

        issues = []
        for issue in data.get("results", []):
            issues.append({
                "test_id": issue.get("test_id"),
                "test_name": issue.get("test_name"),
                "severity": issue.get("issue_severity"),
                "confidence": issue.get("issue_confidence"),
                "line_number": issue.get("line_number"),
                "line_range": issue.get("line_range", []),
                "issue_text": issue.get("issue_text"),
                "cwe": issue.get("issue_cwe", {}),
                "more_info": issue.get("more_info"),
                "code": issue.get("code", "").strip(),
            })

        metrics_raw = data.get("metrics", {}).get("_totals", {})
        metrics = {
            "high": int(metrics_raw.get("SEVERITY.HIGH", 0)),
            "medium": int(metrics_raw.get("SEVERITY.MEDIUM", 0)),
            "low": int(metrics_raw.get("SEVERITY.LOW", 0)),
        }

        return {
            "success": True,
            "result": {
                "filename": filename,
                "issues": issues,
                "metrics": metrics,
                "total_issues": len(issues),
            },
        }

    except SubprocessTimeoutError as exc:
        return {"success": False, "error": str(exc), "result": {}}
    except Exception as exc:
        return {"success": False, "error": str(exc), "result": {}}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def run_dependency_audit(
    requirements_content: str,
    source_filename: str = "requirements.txt",
) -> dict[str, Any]:
    """Run pip-audit against requirements file content to find known CVEs."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".txt",
            delete=False,
            encoding="utf-8",
        ) as tmp:
            tmp.write(requirements_content)
            tmp_path = tmp.name

        result = run_tool(
            ["pip-audit", "-r", tmp_path, "--format", "json", "--progress-spinner", "off"],
            timeout_seconds=120,
        )

        if result.returncode == 127:
            return {
                "success": False,
                "error": "pip-audit is not installed. Run: pip install pip-audit",
                "result": {},
            }

        if not result.stdout.strip():
            return {
                "success": True,
                "result": {
                    "source_filename": source_filename,
                    "vulnerabilities": [],
                    "total_packages_scanned": 0,
                    "vulnerable_count": 0,
                    "note": result.stderr[:500] if result.stderr else "No output",
                },
            }

        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError:
            return {
                "success": False,
                "error": f"Failed to parse pip-audit JSON: {result.stdout[:300]}",
                "result": {},
            }

        vulns = []
        for dep in data.get("dependencies", []):
            for vuln in dep.get("vulns", []):
                vulns.append({
                    "package": dep.get("name"),
                    "installed_version": dep.get("version"),
                    "vulnerability_id": vuln.get("id"),
                    "description": vuln.get("description", ""),
                    "fix_versions": vuln.get("fix_versions", []),
                    "aliases": vuln.get("aliases", []),
                })

        total_packages = len(data.get("dependencies", []))
        vulnerable_packages = len({v["package"] for v in vulns})

        return {
            "success": True,
            "result": {
                "source_filename": source_filename,
                "vulnerabilities": vulns,
                "total_packages_scanned": total_packages,
                "vulnerable_count": vulnerable_packages,
                "total_vulnerabilities": len(vulns),
            },
        }

    except SubprocessTimeoutError as exc:
        return {"success": False, "error": str(exc), "result": {}}
    except Exception as exc:
        return {"success": False, "error": str(exc), "result": {}}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
