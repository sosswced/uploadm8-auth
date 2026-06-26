#!/usr/bin/env python3
"""
Dynamic workflow planner for UploadM8 agents (Level 8).

Inspects changed paths (git diff) and recommends subagent graph + eval modes.

Examples:
  python scripts/agent/dynamic_workflow.py --json
  python scripts/agent/dynamic_workflow.py --staged --json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

LANE_RULES: list[tuple[tuple[str, ...], dict]] = [
    (("frontend/",), {
        "eval_modes": ["frontend", "unit"],
        "rules": ["frontend-static.mdc"],
        "subagents": [{"type": "generalPurpose", "focus": "frontend UI/JS"}],
        "skills": ["uploadm8-frontend-push"],
    }),
    (("routers/", "services/", "core/", "stages/", "api/"), {
        "eval_modes": ["router", "unit"],
        "rules": ["backend-python.mdc"],
        "subagents": [{"type": "explore", "focus": "map API + service flow"}, {"type": "generalPurpose", "focus": "implement backend"}],
        "skills": ["uploadm8-backend-push", "uploadm8-eval-loop"],
    }),
    (("tests/e2e/",), {
        "eval_modes": ["e2e", "overnight"],
        "rules": ["tests-quality.mdc"],
        "subagents": [{"type": "shell", "focus": "run e2e/overnight"}],
        "skills": ["uploadm8-eval-loop"],
    }),
    (("tests/",), {
        "eval_modes": ["unit"],
        "rules": ["tests-quality.mdc"],
        "subagents": [{"type": "generalPurpose", "focus": "fix tests via eval loop"}],
        "skills": ["uploadm8-eval-loop"],
    }),
    ((".cursor/", "scripts/agent/", "AGENTS.md", "docs/agent-stack.md"), {
        "eval_modes": ["unit"],
        "rules": ["uploadm8-core.mdc"],
        "subagents": [{"type": "explore", "focus": "agent stack impact"}],
        "skills": ["uploadm8-agent-orchestrator"],
    }),
]


def git_paths(staged: bool) -> list[str]:
    cmd = ["git", "diff", "--name-only"]
    if staged:
        cmd.append("--cached")
    else:
        cmd.extend(["HEAD"])
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if proc.returncode != 0:
        proc = subprocess.run(["git", "status", "--porcelain"], cwd=str(ROOT), capture_output=True, text=True)
        lines = []
        for line in (proc.stdout or "").splitlines():
            if len(line) > 3:
                lines.append(line[3:].strip())
        return [p.replace("\\", "/") for p in lines]
    return [p.strip().replace("\\", "/") for p in (proc.stdout or "").splitlines() if p.strip()]


def plan(paths: list[str]) -> dict:
    if not paths:
        return {
            "workflow": "idle",
            "eval_modes": ["unit"],
            "subagents": [],
            "skills": ["uploadm8-eval-loop"],
            "parallel": False,
            "message": "No changes detected — run unit eval before ship.",
        }

    eval_modes: list[str] = []
    rules: set[str] = set()
    subagents: list[dict] = []
    skills: set[str] = set()
    lanes: set[str] = set()

    for path in paths:
        for prefixes, cfg in LANE_RULES:
            if any(path.startswith(p) or f"/{p}" in path for p in prefixes):
                lanes.add(prefixes[0].rstrip("/"))
                eval_modes.extend(cfg["eval_modes"])
                rules.update(cfg["rules"])
                subagents.extend(cfg["subagents"])
                skills.update(cfg["skills"])

    if not eval_modes:
        eval_modes = ["unit"]
        skills.add("uploadm8-eval-loop")

    # Dedupe preserving order
    def dedupe(seq: list) -> list:
        seen: set = set()
        out = []
        for item in seq:
            key = json.dumps(item, sort_keys=True) if isinstance(item, dict) else item
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
        return out

    eval_modes = dedupe(eval_modes)
    subagents = dedupe(subagents)

    parallel = len(lanes) >= 2 or (any("frontend" in p for p in paths) and any(p.startswith(("routers/", "services/")) for p in paths))

    workflow = "ascended-loop"
    if parallel:
        workflow = "multi-lane-parallel"
    elif "tests/" in str(paths):
        workflow = "fix-tests"
    elif any(p.startswith(".cursor/") for p in paths):
        workflow = "agent-stack"

    return {
        "workflow": workflow,
        "lanes": sorted(lanes),
        "changed_paths": paths[:50],
        "path_count": len(paths),
        "eval_modes": eval_modes,
        "rules": sorted(rules),
        "subagents": subagents,
        "skills": sorted(skills),
        "parallel": parallel,
        "recommended_commands": [
            f"python scripts/agent/eval_loop.py --mode {m} --json" for m in eval_modes[:3]
        ],
        "slash_commands": ["/ascended-loop" if workflow == "ascended-loop" else f"/{workflow.replace('_', '-')}"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="UploadM8 dynamic workflow planner")
    parser.add_argument("--staged", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    paths = git_paths(args.staged)
    report = plan(paths)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
