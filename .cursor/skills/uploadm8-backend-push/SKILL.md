---
name: uploadm8-backend-push
description: >-
  Stages, commits, and pushes the UploadM8 backend scope (api, core, routers,
  stages, services, scripts, and related root files) to
  https://github.com/sosswced/uploadm8-auth. Use when the user asks to push
  backend to GitHub, sync backend code to uploadm8-auth, or mentions pushing
  these backend folders and files.
---

# UploadM8 backend push to GitHub

Push **only** the backend scope below to `https://github.com/sosswced/uploadm8-auth`. Do not stage or push paths outside this list unless the user explicitly expands scope.

## Remote

- **Repository**: `https://github.com/sosswced/uploadm8-auth`
- **Remote name**: `origin` (verify with `git remote -v` before pushing)

## Push scope

Stage and push **only** these paths (relative to repo root):

| Category | Paths |
|----------|-------|
| Directories | `api/`, `core/`, `docs/`, `jobs/`, `migrations/`, `routers/`, `schemas/`, `scripts/`, `services/`, `stages/`, `tools/` |
| Root files | `app.py`, `telemetry_trill.py`, `worker.py`, `Dockerfile`, `requirements.txt`, `requirements-api.txt`, `requirements-lock.txt`, `runtime.txt` |

Also stage `.gitignore` when it has changes (needed to keep secrets and caches out of the repo).

**Out of scope** — do not stage unless the user explicitly asks:

- `frontend/` and other UI assets
- `.env`, `.env.*`, credentials, API keys, tokens
- `__pycache__/`, `*.pyc`, `.pytest_cache/`, `.venv/`
- Root files not listed above (e.g. `render.yaml`, `.dockerignore`) unless the user adds them

For the full path list, see [paths.md](paths.md).

## Workflow

Copy and track progress:

```
Backend push:
- [ ] 1. Pre-flight checks
- [ ] 2. Clean Python cache artifacts
- [ ] 3. Stage scoped paths only
- [ ] 4. Review diff for secrets and out-of-scope files
- [ ] 5. Commit (only if user asked)
- [ ] 6. Push to origin
- [ ] 7. Verify on GitHub
```

### Step 1: Pre-flight checks

Run in parallel:

```powershell
git status
git remote -v
git branch -vv
```

Confirm:

- `origin` points at `https://github.com/sosswced/uploadm8-auth.git`
- You are on the intended branch (usually `main`)
- No `.env` or secret files appear in changes

### Step 2: Clean Python cache artifacts

Before staging, remove cache junk so it never gets committed. Follow the **remove-python-cache-artifacts** skill (activation keyword: **glooo**), or run the PowerShell cleanup from that skill in repo root.

### Step 3: Stage scoped paths only

Stage each path explicitly — do **not** use `git add .` or `git add -A`.

```powershell
git add `
  api/ core/ docs/ jobs/ migrations/ routers/ schemas/ scripts/ services/ stages/ tools/ `
  app.py telemetry_trill.py worker.py Dockerfile `
  requirements.txt requirements-api.txt requirements-lock.txt runtime.txt `
  .gitignore
```

If a path has no changes, `git add` is harmless. Skip paths that do not exist.

### Step 4: Review diff

```powershell
git diff --cached --stat
git diff --cached
```

Before committing, confirm:

- No `.env`, credentials, or tokens in the staged diff
- No `__pycache__` or `.pyc` files staged
- No unintended `frontend/` or other out-of-scope paths staged

If secrets or cache files are staged, unstage and fix:

```powershell
git reset HEAD -- path/to/file
```

### Step 5: Commit

**Only commit when the user explicitly asks.** Follow the repo's git safety rules:

- Never update git config
- Never skip hooks (`--no-verify`)
- Never force-push to `main`/`master`
- Draft a concise message focused on **why**, not a file list

Example:

```powershell
git commit -m "Update backend pipeline stages, routers, and ML training scripts."
```

For multi-line messages on PowerShell, use a here-string:

```powershell
git commit -m @"
Update backend pipeline stages, routers, and ML training scripts.

"@
```

### Step 6: Push

```powershell
git push -u origin HEAD
```

Use `-u` when the branch has no upstream. Do not force-push unless the user explicitly requests it.

### Step 7: Verify on GitHub

```powershell
gh repo view sosswced/uploadm8-auth --web
gh api repos/sosswced/uploadm8-auth/commits/HEAD --jq '.sha,.commit.message'
```

Or confirm the latest commit on GitHub matches the local push.

## Safety rules

1. **Never commit secrets** — `.env` is gitignored; if it appears in status, do not add it.
2. **Scoped staging only** — backend push means the paths in this skill, not the whole repo.
3. **No destructive git** — no `git push --force`, `git reset --hard`, or config changes unless the user explicitly requests them.
4. **Commit only on request** — stage and show diff by default; commit and push only when asked.

## Related skills

- **remove-python-cache-artifacts** — run before push when `__pycache__` or `.pytest_cache` is present (keyword: **glooo**)
- **creating-pull-requests** — use `gh pr create` when pushing a feature branch instead of directly to `main`

## Examples

**User:** "Push the backend to GitHub"

1. Pre-flight → clean cache → stage scoped paths → show `git diff --cached --stat`
2. Ask whether to commit and push, or commit with a drafted message if they said "push" (implies commit + push)

**User:** "Stage backend changes but don't commit yet"

1. Steps 1–4 only; stop before commit

**User:** "glooo then push backend"

1. Run cache cleanup (glooo)
2. Full workflow through push
