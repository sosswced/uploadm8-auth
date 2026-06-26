---
name: uploadm8-eval-loop
description: >-
  Runs the UploadM8 eval harness and self-heal test failures. Use when tests
  fail, the user asks to fix tests, eval-driven development, Level 8 ascended
  loops, or after implementing backend changes that need verification.
---

# UploadM8 eval loop (Level 8)

Structured test gate for agent self-heal.

## Harness

```powershell
python scripts/agent/eval_loop.py --mode unit --json
python scripts/agent/eval_loop.py --mode frontend --json
python scripts/agent/eval_loop.py --mode full --json
```

## JSON schema

```json
{
  "ok": true,
  "exit_code": 0,
  "mode": "unit",
  "failures": [],
  "failure_count": 0,
  "suggested_command": "python run_tests.py unit"
}
```

## Self-heal loop

Default budget: **5** iterations.

1. Run harness with `--json`
2. If `ok` → done
3. For each failure, read the test file and implementation
4. Fix root cause (never delete tests to green)
5. Re-run harness
6. Stop at budget; report remaining failures

## Mode selection

| Changed paths | Mode |
|---------------|------|
| `routers/`, `services/`, `core/`, `stages/` | `unit` |
| `frontend/` | `frontend` then `unit` if API touched |
| Router structure | `router` then `unit` |

## Integration

- Hooks suggest commands after edits
- `subagentStop` on bugbot chains into this skill
- `/fix-tests` command is the user-facing entry point
