---
name: uploadm8-worktrees
description: >-
  Spawns git worktrees for parallel UploadM8 agent lanes (implement, fix,
  frontend, tests, docs). Use when the user wants parallel branches, multiple
  Cursor windows, git worktrees, or Level 5 degenerate multi-agent workflows.
---

# UploadM8 git worktrees (Level 5)

Parallel agent lanes without branch thrashing.

## When to use

- Multiple features in flight
- One agent implements while another runs tests
- Frontend and backend changes in separate windows

## Spawn

```powershell
.\scripts\agent\worktree_spawn.ps1 -Branch feat/my-change -Lane implement
.\scripts\agent\worktree_spawn_all.ps1 -BranchPrefix feat/my-change
```

Lanes: `implement`, `fix`, `frontend`, `tests`, `docs`.

Worktrees land at `../uploadm8-auth-wt-<lane>/` (sibling to repo root).

## 5-window pattern

| Lane | Branch example | Agent focus |
|------|----------------|-------------|
| implement | `feat/upload-queue` | generalPurpose implementation |
| fix | `fix/r2-guard` | `/fix-tests` loop |
| frontend | `feat/dashboard-ui` | frontend-static rules |
| tests | `test/e2e-upload` | E2E and eval harness |
| docs | `docs/agent-stack` | AGENTS.md, docs/ |

## Cleanup

```powershell
git worktree list
git worktree remove ../uploadm8-auth-wt-implement
git branch -d feat/my-change   # after merge
```

## Rules

- Each worktree is an independent Cursor window — one agent per lane.
- Merge via PR; never force-push main from a worktree.
- Backend push skill applies per worktree checkout.
