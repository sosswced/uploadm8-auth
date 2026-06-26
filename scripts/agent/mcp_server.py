#!/usr/bin/env python3
"""
UploadM8 custom MCP server — exposes agent tooling to Cursor (Level 8).

Requires: pip install mcp  (see requirements-agent.txt)

Tools: eval_run, self_heal, workflow_plan, ci_status, multi_repo_status, trill_run

Start via .cursor/mcp.json:
  "uploadm8-agent": { "command": "python", "args": ["scripts/agent/mcp_server.py"] }
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AGENT = Path(__file__).resolve().parent


def run_script(script: str, *args: str) -> str:
    proc = subprocess.run(
        [sys.executable, str(AGENT / script), *args, "--json"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return (proc.stdout or proc.stderr or "").strip()


def build_server():
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        print(
            "UploadM8 MCP server requires: pip install mcp",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    mcp = FastMCP("uploadm8-agent")

    @mcp.tool()
    def eval_run(mode: str = "unit") -> str:
        """Run UploadM8 eval harness (unit, frontend, router, full). Returns JSON."""
        return run_script("eval_loop.py", "--mode", mode)

    @mcp.tool()
    def self_heal(mode: str = "unit", budget: int = 5) -> str:
        """Run self-heal eval iterations. Returns JSON with failure history."""
        return run_script("self_heal.py", "--mode", mode, "--budget", str(budget))

    @mcp.tool()
    def workflow_plan(staged: bool = False) -> str:
        """Recommend dynamic agent workflow from git diff. Returns JSON."""
        args = ["--staged"] if staged else []
        proc = subprocess.run(
            [sys.executable, str(AGENT / "dynamic_workflow.py"), *args, "--json"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        return (proc.stdout or "").strip()

    @mcp.tool()
    def ci_status(pr: int = 0) -> str:
        """Get GitHub PR checks as JSON via gh CLI."""
        args = ["--pr", str(pr)] if pr else []
        proc = subprocess.run(
            [sys.executable, str(AGENT / "ci_status.py"), *args, "--json"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        return (proc.stdout or proc.stderr or "").strip()

    @mcp.tool()
    def multi_repo_status() -> str:
        """Backend + frontend repo dirty state for multi-repo orchestration."""
        return run_script("multi_repo_status.py")

    @mcp.tool()
    def trill_run(mode: str = "audit") -> str:
        """TRILL master orchestrator — full stack JSON report (audit, pre-ship, heal, overnight, full)."""
        allowed = ("audit", "pre-ship", "heal", "overnight", "full")
        m = mode if mode in allowed else "audit"
        return run_script("trill.py", "--mode", m)

    return mcp


def main() -> None:
    build_server().run()


if __name__ == "__main__":
    main()
