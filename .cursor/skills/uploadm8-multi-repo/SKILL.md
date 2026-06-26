---
name: uploadm8-multi-repo
description: >-
  Cross-repo orchestration for uploadm8-auth backend and uploadm8-frontend.
  Use when both repos change, multi-repo ship, dual-repo deploy, or agents
  managing agents across backend and frontend PRs.
---

# UploadM8 multi-repo orchestration

Dual-repo deploy: backend → `sosswced/uploadm8-auth`, frontend → `sosswced/uploadm8-frontend`.

## Status check

```powershell
python scripts/agent/multi_repo_status.py --json
```

## Orchestration graph

```
Manager agent (this session)
    │
    ├─► Backend lane → uploadm8-backend-push skill → PR A
    │
    ├─► Frontend lane → uploadm8-frontend-push skill → PR B
    │
    └─► uploadm8-ascended-ci on each PR
```

## Parallel pattern

When both repos need work:

1. Spawn worktrees: `implement` + `frontend` lanes
2. Or launch two background `generalPurpose` subagents (one per repo scope)
3. Manager synthesizes eval results before ship

## Ship order

1. Backend API changes first (if frontend depends on new endpoints)
2. Frontend sync second
3. Ascended CI on both PRs

## Rules

- Never mix scopes in one commit
- Never `git add .` on backend push
- Commit/push only when user explicitly asks
- Frontend clone default: `~/Dev/uploadm8-frontend` or sibling `../uploadm8-frontend`
