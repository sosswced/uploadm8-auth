---
name: uploadm8-ultrathink
description: >-
  Deep planning mode for UploadM8: exhaustive explore, architecture review,
  risk matrix, and phased plan before any code. Use when the user says
  ultrathink, deep plan, architecture review, or complex multi-system changes.
---

# UploadM8 ultrathink

**Context is everything.** Before writing code, build a complete map.

## Phase 0 — Context load

1. Read `AGENTS.md` and `docs/agent-stack.md`
2. Run `python scripts/agent/dynamic_workflow.py --json` for lane hints
3. Read `routers/README.md` if touching HTTP surface

## Phase 1 — Parallel discovery (single turn)

Launch **three** readonly subagents in parallel:

| Subagent | Task |
|----------|------|
| `explore` (very thorough) | Map all files, call chains, test coverage |
| `explore` (medium) | Find similar patterns already in repo |
| `bugbot` (readonly) | Review existing diff if any |

## Phase 2 — Architecture brief

Deliver to user (do not implement yet):

1. **Problem statement** — one paragraph
2. **Touch map** — tables of files/modules
3. **Risks** — data loss, auth, billing, pipeline, dual-repo
4. **Test strategy** — eval modes + new tests needed
5. **Phased plan** — 3–7 steps with rollback notes
6. **Subagent graph** — which agents run in which order after approval

## Phase 3 — Gate

Stop and ask: **"Proceed with implementation?"**

Only continue if user approves or said "fully autonomous" / "ultrathink and implement".

## Integration

- Entry: `/ultrathink` slash command
- Chains into `uploadm8-agent-orchestrator` for execution
- Never skip eval harness after implementation
