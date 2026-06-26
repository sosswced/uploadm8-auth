---
name: uploadm8-dynamic-workflows
description: >-
  Picks subagent graph and eval modes from git diff. Use for dynamic workflows,
  agents managing agents, workflow planning, or when unsure which skill to run.
---

# UploadM8 dynamic workflows (Level 8)

Let the diff decide the agent graph.

## Planner

```powershell
python scripts/agent/dynamic_workflow.py --json
python scripts/agent/dynamic_workflow.py --staged --json
```

## Output fields

| Field | Use |
|-------|-----|
| `workflow` | Which slash command to run |
| `eval_modes` | Ordered eval_loop modes |
| `subagents` | Parallel or sequential Task types |
| `skills` | Skills to invoke |
| `parallel` | true → launch subagents in one turn |

## Decision tree

```
dynamic_workflow.py
    │
    ├─ workflow=fix-tests ──► /fix-tests
    ├─ workflow=multi-lane-parallel ──► parallel subagents + worktrees
    ├─ workflow=agent-stack ──► uploadm8-agent-orchestrator
    └─ workflow=ascended-loop ──► /ascended-loop
```

## MCP

If `uploadm8-agent` MCP is enabled, call `workflow_plan` tool instead.

## Manager pattern

One manager agent:
1. Runs dynamic_workflow
2. Spawns worker subagents per `subagents` array
3. Collects results → eval_loop → ascended CI
