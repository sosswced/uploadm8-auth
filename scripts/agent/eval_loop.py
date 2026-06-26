#!/usr/bin/env python3
"""
UploadM8 eval harness for agent self-heal loops (Level 8).

Runs targeted test modes and returns structured JSON for agents.

Examples:
  python scripts/agent/eval_loop.py
  python scripts/agent/eval_loop.py --mode unit --json
  python scripts/agent/eval_loop.py --mode full
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

MODES: dict[str, list[str]] = {
    "unit": ["unit", "-q", "--tb=line", "--maxfail=5"],
    "frontend": ["frontend-lint"],
    "router": ["router-lint"],
    "full": ["unit", "-q", "--tb=line", "--maxfail=10"],
}

FAILURE_RES = [
    re.compile(
        r"^FAILED\s+(?P<file>tests[/\\][^\s:]+\.py)::(?P<nodeid>[^\s]+)",
        re.MULTILINE,
    ),
    re.compile(
        r"^(?P<file>tests[/\\][^\s:]+\.py)::(?P<nodeid>[^\s]+)\s+FAILED",
        re.MULTILINE,
    ),
]
ERROR_RE = re.compile(
    r"^ERROR\s+(?P<file>tests[/\\][^\s:]+\.py)::(?P<nodeid>[^\s]+)",
    re.MULTILINE,
)
SUMMARY_RE = re.compile(
    r"(?P<failed>\d+)\s+failed.*?(\d+)\s+passed.*?(\d+)\s+errors",
    re.IGNORECASE,
)


def run_mode(mode: str) -> tuple[int, str]:
    runner = ROOT / "run_tests.py"
    cmd = [sys.executable, str(runner)] + MODES.get(mode, MODES["unit"])
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, output


def parse_failures(output: str) -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []
    seen: set[str] = set()
    for pattern in (*FAILURE_RES, ERROR_RE):
        for match in pattern.finditer(output):
            key = f"{match.group('file')}::{match.group('nodeid')}"
            if key in seen:
                continue
            seen.add(key)
            failures.append({
                "file": match.group("file").replace("\\", "/"),
                "nodeid": match.group("nodeid"),
            })
    return failures


def parse_summary(output: str) -> dict[str, int] | None:
    match = SUMMARY_RE.search(output)
    if not match:
        return None
    return {
        "failed": int(match.group(1)),
        "passed": int(match.group(2)),
        "errors": int(match.group(3)),
    }


def suggest_command(mode: str, failures: list[dict[str, str]]) -> str:
    if not failures:
        return f"python run_tests.py {mode}"
    first = failures[0]["nodeid"].split("[")[0].split("::")[-1]
    if first.startswith("test_"):
        return f"python run_tests.py {mode} -k {first}"
    return f"python run_tests.py {mode}"


def main() -> int:
    parser = argparse.ArgumentParser(description="UploadM8 eval harness")
    parser.add_argument("--mode", choices=list(MODES), default="unit")
    parser.add_argument("--json", action="store_true", help="Print JSON report only")
    args = parser.parse_args()

    code, output = run_mode(args.mode)
    failures = parse_failures(output)
    summary = parse_summary(output)
    report = {
        "ok": code == 0,
        "exit_code": code,
        "mode": args.mode,
        "summary": summary,
        "failures": failures[:20],
        "failure_count": len(failures),
        "suggested_command": suggest_command(args.mode, failures),
    }

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(json.dumps(report, indent=2))
        if not report["ok"] and output.strip():
            print("\n--- pytest output (tail) ---")
            lines = output.strip().splitlines()
            tail = lines[-40:] if len(lines) > 40 else lines
            print("\n".join(tail))

    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
