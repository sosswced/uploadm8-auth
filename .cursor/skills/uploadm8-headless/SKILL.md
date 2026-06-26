---
name: uploadm8-headless
description: >-
  Headless and background agent patterns for long-running UploadM8 tasks.
  Use for background agents, run_in_background, unattended eval, overnight
  runs, or shell subagents that continue while parent works.
---

# UploadM8 headless agents (Level 7)

Run long work without blocking the parent agent.

## Task tool pattern

```
Task(
  subagent_type="shell",
  run_in_background=true,
  description="overnight e2e",
  prompt="Run python run_tests.py overnight. Summarize failures."
)
```

## Headless eval

```powershell
python scripts/agent/self_heal.py --mode unit --budget 5 --json
```

Parent agent: launch in background shell, continue other work, act on notification.

## Overnight

```powershell
python run_tests.py overnight
```

Use background `shell` subagent. Triage in next session with `/fix-tests`.

## Explore at scale

```powershell
Task(subagent_type="explore", run_in_background=true, readonly=true, ...)
```

## Rules

- Always set unique `description` for tracking
- On completion notification → run `eval_loop` before ship
- Do not commit unless user asked
