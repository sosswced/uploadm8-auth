---
name: "333"
description: >-
  UploadM8 full deploy shortcut: removes Python cache artifacts, pushes backend
  to sosswced/uploadm8-auth, then syncs and pushes frontend to
  sosswced/uploadm8-frontend. Use when the user says /333, 333, triple push,
  deploy both repos, or glooo then push backend and frontend.
disable-model-invocation: true
---

# /333 — Clean cache + backend push + frontend push

One-shot UploadM8 deploy. Runs three child skills **in order**; do not skip or reorder unless the user narrows scope (e.g. "333 backend only").

## Child skills (read and follow each)

| Step | Skill | Location |
|------|-------|----------|
| 1 | **remove-python-cache-artifacts** | `~/.cursor/skills/remove-python-cache-artifacts/SKILL.md` |
| 2 | **uploadm8-backend-push** | `.cursor/skills/uploadm8-backend-push/SKILL.md` |
| 3 | **uploadm8-frontend-push** | `.cursor/skills/uploadm8-frontend-push/SKILL.md` |

Path details: [paths-backend.md](paths-backend.md), [paths-frontend.md](paths-frontend.md)

## Activation

Treat any of these as **full /333** (cleanup + both pushes + commit unless user says otherwise):

- `/333`, `333`, `triple push`, `deploy both repos`
- `glooo` + push backend and frontend in the same message
- "clean cache and push everything"

**Partial runs** — only when the user explicitly limits scope:

| User says | Run |
|-----------|-----|
| `333 backend` / `333 backend only` | Steps 1–2 |
| `333 frontend` / `333 frontend only` | Step 3 only (skip cache unless user also said glooo) |
| `333 no push` / `333 dry run` | Steps 1–3 through review/diff only; no commit or push |

## Checklist

```
/333 deploy:
- [ ] 1. Cache cleanup (remove-python-cache-artifacts)
- [ ] 2. Backend pre-flight → stage → diff → commit → push (uploadm8-backend-push)
- [ ] 3. Frontend pre-flight → robocopy → diff → commit → push (uploadm8-frontend-push)
- [ ] 4. Verify both remotes (gh or git log)
```

## Step 1 — Cache cleanup

Follow **remove-python-cache-artifacts**. `/333` **implies glooo** — delete `__pycache__/`, `.pytest_cache/`, `*.pyc`, `*.pyo` in `uploadm8-auth` root without a separate confirmation.

PowerShell (repo root `uploadm8-auth`):

```powershell
Get-ChildItem -Path . -Recurse -Directory -Filter __pycache__ -Force -ErrorAction SilentlyContinue |
  Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Get-ChildItem -Path . -Recurse -Directory -Filter .pytest_cache -Force -ErrorAction SilentlyContinue |
  Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Get-ChildItem -Path . -Recurse -File -Include *.pyc,*.pyo -Force -ErrorAction SilentlyContinue |
  Remove-Item -Force -ErrorAction SilentlyContinue
git status --short
```

## Step 2 — Backend push (`uploadm8-auth`)

Follow **uploadm8-backend-push** completely:

- Repo: `C:\Users\Earl\Dev\uploadm8-auth`
- Remote: `https://github.com/sosswced/uploadm8-auth`
- Stage **scoped paths only** — never `git add .`
- `/333` means **commit and push** unless user said dry run / no push
- Never stage `frontend/`, `.env`, or cache artifacts

## Step 3 — Frontend push (`uploadm8-frontend`)

Follow **uploadm8-frontend-push** completely:

- Source: `uploadm8-auth/frontend/`
- Clone: `C:\Users\Earl\Dev\uploadm8-frontend`
- Remote: `https://github.com/sosswced/uploadm8-frontend`
- **Never** `git stash push -u` on the auth repo
- **Never** `git add .` in either repo
- robocopy → explicit stage → commit → `git pull --rebase origin main` → push

## Safety (non-negotiable)

1. Two repos, two pushes — never one combined commit across repos
2. No secrets in staged diffs (`.env`, credentials, tokens)
3. No `git push --force` on `main` unless user explicitly requests
4. No git config changes, no `--no-verify`
5. Report both commit SHAs (or errors) when finished

## Completion report

When done, summarize:

```markdown
## /333 complete

| Step | Repo | Result |
|------|------|--------|
| Cache | uploadm8-auth | [clean / issues] |
| Backend | sosswced/uploadm8-auth | [sha + message / skipped] |
| Frontend | sosswced/uploadm8-frontend | [sha + message / skipped] |
```

## Examples

**User:** `/333`

1. glooo cleanup in auth repo
2. Backend: stage scoped paths → commit → push `origin`
3. Frontend: robocopy → commit → push `origin main`

**User:** `333 backend only`

1. Cache cleanup
2. Backend push only

**User:** `333 dry run`

1. Cleanup + stage + show diffs in both repos; stop before commit
