---
name: uploadm8-agent-orchestrator
description: >-
  Orchestrates multi-phase UploadM8 agent teams: plan (explore), implement,
  eval (eval_loop), review (bugbot), optional ship. Use for complex features,
  Level 7 fallen overnight-style runs, agent teams, or when the user asks for
  ascended or orchestrated multi-agent workflows.
---

# UploadM8 agent orchestrator (Level 7)

Structured agent team for non-trivial work.

## Phases

Copy and track:

```
Orchestrator:
- [ ] 1. Plan (explore, readonly)
- [ ] 2. Implement (scoped diff)
- [ ] 3. Eval (eval_loop.py)
- [ ] 4. Review (bugbot, readonly)
- [ ] 5. Ship (only if user asked)
```

### Phase 1 — Plan

Launch `explore` subagent (medium thoroughness):

- Map files to touch
- List test targets (`run_tests.py` mode)
- Output 3–5 step plan with risks

**Stop and show plan to user** unless they said "fully autonomous".

### Phase 2 — Implement

Execute plan with minimal diff. Match `AGENTS.md` contracts.

### Phase 3 — Eval (mandatory)

```powershell
python scripts/agent/eval_loop.py --mode unit --json
```

If `ok: false`, run fix-tests logic (max 3 iterations):

1. Parse `failures`
2. Fix root cause
3. Re-run eval

### Phase 4 — Review

Launch `bugbot` subagent (readonly) on branch diff or uncommitted changes.

Address **critical** findings before ship.

### Phase 5 — Ship

Only when user explicitly requested push/commit:

- Backend → `uploadm8-backend-push` skill
- Frontend → `uploadm8-frontend-push` skill

## Background mode

For long E2E or explore phases, use Task with `run_in_background: true`. Parent continues; act on notification.

## Overnight variant

Replace Phase 3 eval with:

```powershell
python run_tests.py overnight
```

Run in background shell subagent. Triage failures next session.

## Parallel shortcut

Phases 1 + 4 can run in parallel at start via `/parallel-audit` if reviewing existing work.
