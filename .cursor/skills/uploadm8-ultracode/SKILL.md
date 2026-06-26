---
name: uploadm8-ultracode
description: >-
  Eval-driven development where tests are the reward signal. Agents propose
  minimal diffs, run eval_loop, iterate until green, then optional architecture
  PR. Use for ultracode, eval-as-reward, or beyond Level 8 architecture loops.
---

# UploadM8 ultracode (beyond Level 8)

Treat `eval_loop.py` output as the **reward signal**. Code that doesn't improve evals gets reverted.

## Loop

```
propose diff → eval_loop → score ok/fail → patch or revert → repeat
```

## Commands

```powershell
python scripts/agent/eval_loop.py --mode unit --json
python scripts/agent/self_heal.py --mode unit --budget 5 --json
python scripts/agent/dynamic_workflow.py --json
```

## Rules

1. **One hypothesis per iteration** — small diffs only
2. **Never delete tests** to green — fix behavior
3. **Track score** — report iteration count and failure delta
4. After 3 stalls on same failure → `explore` subagent deep-dive
5. Green eval → optional `bugbot` → user decides ship

## Architecture PRs

When eval is green and user asked for architecture work:

1. Document trade-offs in PR body
2. Link eval JSON showing green
3. Use `uploadm8-ascended-ci` for PR checks

## Ultracode vs fix-tests

| Mode | Scope |
|------|-------|
| `/fix-tests` | Repair regressions only |
| `/ultracode` | Feature + tests + eval-driven iteration |
