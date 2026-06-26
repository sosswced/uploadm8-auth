---
name: uploadm8-ascended-ci
description: >-
  CI/CD agent workflow for UploadM8: check PR status with gh, triage failing
  checks, fix, re-run eval loop, push. Use when CI fails, PR checks are red,
  Level 8 ascended CI/CD, or multi-repo ship after backend and frontend changes.
---

# UploadM8 ascended CI (Level 8)

Agents manage the path from green local evals to green PR checks.

## Preconditions

- `gh` authenticated
- Branch pushed with open PR (or user will create one)
- Local eval green: `python scripts/agent/eval_loop.py --mode unit --json`

## Workflow

```
Ascended CI:
- [ ] 1. Local eval green
- [ ] 2. gh pr checks / gh run list
- [ ] 3. Triage failures (ci-investigator subagent per check)
- [ ] 4. Fix + local eval
- [ ] 5. Push (only if user asked)
- [ ] 6. Re-verify checks
```

### Step 1 — Local gate

```powershell
python scripts/agent/eval_loop.py --mode full --json
```

Do not proceed if local eval fails.

### Step 2 — PR checks

```powershell
gh pr checks
gh run list --limit 5
```

### Step 3 — Triage

For each failing check, launch `ci-investigator` subagent with check name and PR URL.

### Step 4 — Fix loop

Apply fixes → `eval_loop` → repeat until local green.

### Step 5 — Push

Only on user request:

```powershell
git push -u origin HEAD
```

### Step 6 — Verify

```powershell
gh pr checks --watch
```

## Multi-repo ship

When both backend and frontend changed in one feature:

1. Backend PR → `uploadm8-backend-push` skill
2. Frontend PR → `uploadm8-frontend-push` skill
3. Run ascended CI on each PR independently

## Rules

- Never force-push main
- Never skip hooks
- Commit only when user explicitly asks
