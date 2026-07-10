#!/usr/bin/env python3
"""
Register agent-built tooling in scripts/agent/tool_registry.json.

Examples:
  python scripts/agent/register_tool.py --name my_tool --path scripts/agent/my_tool.py --purpose "Does X"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = Path(__file__).resolve().parent / "tool_registry.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Register agent-built tool")
    parser.add_argument("--name", required=True)
    parser.add_argument("--path", required=True)
    parser.add_argument("--purpose", required=True)
    parser.add_argument("--skill", default="")
    args = parser.parse_args()

    data = json.loads(REGISTRY.read_text(encoding="utf-8"))
    tools = data.setdefault("tools", [])
    entry = {
        "name": args.name,
        "path": args.path.replace("\\", "/"),
        "purpose": args.purpose,
    }
    if args.skill:
        entry["skill"] = args.skill

    tools = [t for t in tools if t.get("name") != args.name]
    tools.append(entry)
    data["tools"] = sorted(tools, key=lambda t: t["name"])
    REGISTRY.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"Registered {args.name} -> {REGISTRY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
