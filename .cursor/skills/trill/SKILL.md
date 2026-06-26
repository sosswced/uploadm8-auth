---
name: trill
description: >-
  TRILL master orchestrator for UploadM8: runs the full agent stack — route,
  parallel audit, eval, self-heal, overnight, multi-repo, CI, optional ship.
  Use when the user says TRILL, /trill, go live, push, run everything, full
  stack, or wants all agent features in one autonomous loop.
---

# TRILL — master orchestrator

**T**riage · **R**oute · **I**terate · **L**aunch · **L**ive

One entry point for the entire UploadM8 agent stack. TRILL runs what scripts can; the agent runs what needs intelligence (audit, fix, ship).

## When to invoke

- User says: TRILL, `/trill`, "run everything", "go live", "push when green"
- Pre-ship gate before backend + frontend deploy
- Nightly or end-of-session full health check
- After big feature — prove green across eval, audit, CI, both repos

## Modes

| Mode | Script flag | What runs |
|------|-------------|-----------|
| **audit** (default) | `--mode audit` | workflow + evals + repos + CI |
| **pre-ship** | `--mode pre-ship` | audit + frontend + router + unit + full eval |
| **heal** | `--mode heal` | audit + self_heal (5 eval iterations) |
| **overnight** | `--mode overnight` | audit + recommends `/overnight` |
| **full** | `--mode full` | all eval modes + overnight recommendation |

```powershell
python scripts/agent/trill.py --json
python scripts/agent/trill.py --mode pre-ship --json
python scripts/agent/trill.py --mode heal --budget 5 --json
```

Parse `actions[]` in the JSON — each item is the next agent step.

---

## TRILL pipeline (agent execution)

Copy and track:

```
TRILL:
- [ ] 0. Bootstrap — MCP + hooks + trill.py JSON
- [ ] 1. Route — dynamic_workflow.py (already in trill report)
- [ ] 2. Parallel audit — /parallel-audit (explore + bugbot)
- [ ] 3. Eval — eval_loop per workflow; /fix-tests if red
- [ ] 4. Review chain — bugbot → subagentStop → eval (automatic via hooks)
- [ ] 5. Overnight — /headless-eval or /overnight if E2E touched
- [ ] 6. Multi-repo — multi_repo_status → /multi-repo-ship if dirty
- [ ] 7. CI — ci_status → uploadm8-ascended-ci if PR exists
- [ ] 8. Ship — only if user explicitly asked
```

### Phase 0 — Bootstrap

1. Run `python scripts/agent/trill.py --json` (or mode matching user intent)
2. Confirm `env.python_ok`, `stack` files present
3. Enable MCP in Cursor: Sentry, HuggingFace (optional), **uploadm8-agent**
4. Optional: `pip install -r requirements-agent.txt`

### Phase 1 — Route

Use `workflow` from trill JSON:
- `parallel: true` → launch subagents in **one turn**
- Follow `slash_commands[0]` if not continuing full TRILL

### Phase 2 — Parallel audit (mandatory before ship)

Launch in **one turn**:
1. **explore** (quick) — map uncommitted changes, risk areas
2. **bugbot** (readonly) — severity-ranked findings

Synthesize: critical / suggestions / test gaps. **Do not ship with critical open.**

### Phase 3 — Eval + heal

For each mode in `workflow.eval_modes`:
```powershell
python scripts/agent/eval_loop.py --mode <mode> --json
```

If any `ok: false` → **/fix-tests** (max 5 iterations). Never delete tests to green.

If heal mode: `self_heal` section in trill JSON shows iteration history.

### Phase 4 — Review chain (hooks)

After bugbot/security-review/ci-investigator completes, `subagentStop` hook chains eval. Do not skip.

### Phase 5 — Overnight (when warranted)

If workflow touches `tests/e2e/`, `frontend/`, or user said overnight:
- Background: `/headless-eval` or Task shell `run_in_background: true`
- Command: `python run_tests.py overnight`
- Triage failures → `/fix-tests` next session

### Phase 6 — Multi-repo

From trill `repos`:
- `both_dirty` → `/multi-repo-ship` order: backend first if API changed
- Use `uploadm8-backend-push` + `uploadm8-frontend-push` skills

### Phase 7 — CI

```powershell
python scripts/agent/ci_status.py --json
```

If failing → `ci-investigator` per check → fix → eval → push (user asked only)

### Phase 8 — Ship

**Only when user explicitly said push/ship/go live in this session.**

1. `/ship-backend` if backend dirty
2. `/ship-frontend` if frontend dirty
3. Re-run `ci_status` — `gh pr checks --watch`

---

## TRILL variants

| User says | TRILL mode | Then |
|-----------|------------|------|
| "check everything" | audit | parallel-audit if eval green |
| "make it green" | heal | /fix-tests until ok |
| "ready to push" | pre-ship | phases 2–7, ship only if asked |
| "run overnight" | overnight + /overnight | background E2E |
| big new feature | ultrathink first → TRILL pre-ship | plan before code |

---

## MCP

Call `trill_run` on **uploadm8-agent** MCP, or tools individually:
`eval_run`, `self_heal`, `workflow_plan`, `ci_status`, `multi_repo_status`

---

## Unattended (Cursor Automations)

Copy prompt from `.cursor/automations/trill.md` — nightly TRILL audit without an open session.

---

## Parallel lanes

For multi-lane features during TRILL:
```powershell
.\scripts\agent\worktree_spawn_all.ps1 -BranchPrefix feat/trill-<name>
```
Open 5 Cursor windows — see `/five-windows`.

---

## Go live checklist

See `paths-live.md` in this skill folder for exact git paths to push agent stack to GitHub.

---

## Rules

- Never force-push main
- Never commit `.env` or secrets
- Commit/push only when user explicitly asks
- TRILL reports red → fix before ship
- Extend stack via `register_tool.py` + `uploadm8-agent-tooling`
