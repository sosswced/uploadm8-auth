# Orchestrate — agent team workflow

Follow **uploadm8-agent-orchestrator** skill with dynamic routing.

## Start

```powershell
python scripts/agent/dynamic_workflow.py --json
```

Use output to pick parallel vs sequential subagents.

## Phases

- [ ] Plan (explore, readonly) — skip if user gave detailed spec
- [ ] Implement (minimal diff)
- [ ] Eval — `python scripts/agent/eval_loop.py --mode unit --json`
- [ ] Review — bugbot readonly
- [ ] Ship — only if user asked (`/ship-backend`, `/ship-frontend`)

## Parallel shortcut

If `parallel: true` in workflow JSON, launch subagents in **one turn**.

Report phase boundary summaries. Stop if eval red after budget.
