---
name: uploadm8-routines
description: >-
  Cursor Automations and recurring agent routines for UploadM8. Use for
  scheduled runs, nightly tests, push-to-main triage, /loop patterns, or
  unsupervised overnight agent workflows.
---

# UploadM8 routines (Level 7)

Agents working while you sleep — via Cursor Automations or in-session `/loop`.

## Cursor Automations (Settings → Automations)

| Trigger | Action |
|---------|--------|
| Push to `main` | Run `python scripts/agent/eval_loop.py --mode full --json`; comment on PR if red |
| Nightly cron | `python run_tests.py overnight` |
| PR opened | `python scripts/agent/ci_status.py --json` |

## In-session loop

Use `/loop` command — see `.cursor/commands/loop.md`.

Examples:
- `/loop 30m python scripts/agent/eval_loop.py --mode unit --json`
- `/loop check CI until green`

## Overnight agent team

1. Background shell: `python run_tests.py overnight`
2. On failure → queue `/fix-tests` for next session
3. Optional: Sentry MCP triage for production errors

## Unsupervised checklist

```
- [ ] eval_loop green locally
- [ ] overnight marker suite (if E2E touched)
- [ ] bugbot on diff
- [ ] ci_status green (if PR exists)
- [ ] user asked before commit/push
```

## Headless entry

`/headless-eval` slash command for background self-heal driver.
