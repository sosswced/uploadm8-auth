---
name: uploadm8-agent-tooling
description: >-
  Agent-built tooling registry under scripts/agent/. Use when adding repeatable
  agent scripts, registering new tools, or extending the MCP server.
---

# UploadM8 agent-built tooling

Agents may add scripts when a workflow repeats twice.

## Registry

`scripts/agent/tool_registry.json` — canonical list of agent tools.

## Register new tool

```powershell
python scripts/agent/register_tool.py --name my_tool --path scripts/agent/my_tool.py --purpose "What it does" --skill uploadm8-eval-loop
```

Then add a skill reference in `.cursor/skills/<skill>/SKILL.md` if needed.

## MCP exposure

After adding a tool, extend `scripts/agent/mcp_server.py` with a `@mcp.tool()` wrapper.

## Conventions

- Scripts live in `scripts/agent/`
- Accept `--json` for structured agent output
- Exit 0 on success, 1 on failure
- No secrets in scripts — read from `.env` via `run_tests.py` pattern
- Run `python run_tests.py unit` if you add tests for the tool

## Existing tools

See `tool_registry.json` for eval_loop, self_heal, dynamic_workflow, ci_status, multi_repo_status, worktree_spawn.
