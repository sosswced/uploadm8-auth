---
name: uploadm8-frontend-push
description: >-
  Syncs UploadM8 static UI from uploadm8-auth/frontend to the dedicated
  uploadm8-frontend GitHub repo (https://github.com/sosswced/uploadm8-frontend),
  then commits and pushes. Use when the user asks to push frontend, sync HTML/JS/CSS
  to uploadm8-frontend, or deploy the frontend folder to GitHub.
---

# UploadM8 frontend push to GitHub

Push **only** the static UI from `frontend/` in the auth monorepo to the **separate** frontend repository. Do not push `frontend/` through `uploadm8-auth` unless the user explicitly asks for that too.

## Repositories

| Role | Path / URL |
|------|------------|
| **Source** (edit here) | `{auth-repo}/frontend/` — e.g. `C:\Users\Earl\Dev\uploadm8-auth\frontend` |
| **Push target** (clone) | `{frontend-repo}/` — e.g. `C:\Users\Earl\Dev\uploadm8-frontend` |
| **Remote** | `https://github.com/sosswced/uploadm8-frontend` (`origin`) |

The frontend clone root **is** the site tree (`dashboard.html`, `js/`, `css/`, …). Do **not** nest another `frontend/` folder inside the clone.

For path lists and copy exclusions, see [paths.md](paths.md).

## Critical safety rules

1. **Never `git stash push -u` on the auth monorepo** before a frontend push. `-u` stashes **untracked** `frontend/` and can empty the source folder locally.
2. **Never `git add .` or `git add -A`** in either repo for this workflow.
3. **Never commit** `.env`, `*.pem`, `*-credentials.json`, or `node_modules/`.
4. **Do not stage `frontend/` in `uploadm8-auth`** when the user only asked to push to `uploadm8-frontend`.
5. **Commit and push only when the user explicitly asks.**

## Workflow

```
Frontend push:
- [ ] 1. Pre-flight (source has files, clone exists, remotes correct)
- [ ] 2. Sync auth/frontend → frontend clone (exclude secrets + node_modules)
- [ ] 3. Review diff in frontend clone
- [ ] 4. Commit (only if user asked)
- [ ] 5. Push origin
- [ ] 6. Verify on GitHub
```

### Step 1: Pre-flight

Run in the **auth** repo:

```powershell
cd C:\Users\Earl\Dev\uploadm8-auth
(Get-ChildItem frontend -Filter *.html -ErrorAction SilentlyContinue | Measure-Object).Count
git status --short frontend | Select-Object -First 20
```

Confirm:

- HTML count is **> 0** (if 0, restore from stash: `git checkout "stash@{0}^3" -- frontend/` then re-run pre-flight)
- No accidental `git add frontend/` staged in auth unless user wants monorepo tracking

In the **frontend** clone:

```powershell
cd C:\Users\Earl\Dev\uploadm8-frontend
git remote -v
git branch -vv
git status --short | Select-Object -First 30
```

If the clone is missing:

```powershell
cd C:\Users\Earl\Dev
git clone https://github.com/sosswced/uploadm8-frontend.git
```

### Step 2: Sync source → frontend clone

Copy from auth monorepo into the frontend clone root. Default: update/add files; **do not** delete extra files in the clone (`/E` only).

```powershell
$src = "C:\Users\Earl\Dev\uploadm8-auth\frontend"
$dst = "C:\Users\Earl\Dev\uploadm8-frontend"

robocopy $src $dst /E /XD node_modules .git /XF .env .env.* *.pem *-credentials.json credentials.json desktop.ini /NFL /NDL /NJH /NJS /nc /ns /np
```

If the user wants an **exact mirror** (clone matches source; removes files deleted in source):

```powershell
robocopy $src $dst /MIR /XD node_modules .git /XF .env .env.* *.pem *-credentials.json credentials.json desktop.ini /NFL /NDL /NJH /NJS /nc /ns /np
```

`robocopy` exit codes 0–7 mean success (copied or nothing to copy). Codes **≥ 8** indicate errors — inspect output before committing.

### Step 3: Review diff

```powershell
cd C:\Users\Earl\Dev\uploadm8-frontend
git status --short
git diff --stat
```

Before commit, confirm:

- No `.env`, credentials, or `node_modules/` in the diff
- HTML/JS/CSS changes match what the user expected
- No nested `frontend/frontend/` paths appeared (if so, fix sync paths)

Stage **explicitly** — never `git add .`:

```powershell
git add *.html *.js *.css _redirects .gitignore
git add js/ css/ images/ billing/ compare/ 2>$null
git add -u
```

Or stage known subtrees:

```powershell
git add js/ css/ images/ billing/ compare/ app.js billing.js wallet-tokens.js shared-sidebar.js upgrade-modal.js
git add -u -- *.html *.js *.css _redirects .gitignore
```

Re-run `git diff --cached --stat` and scan for secrets.

### Step 4: Commit

Only when the user asked to push/commit. PowerShell-friendly message:

```powershell
git commit -m "Update frontend pages, assets, and client scripts."
```

### Step 5: Push

```powershell
git pull --rebase origin main
git push -u origin HEAD
```

If push is rejected:

- Run `git pull --rebase origin main` (no `git stash -u` on the auth repo)
- Resolve conflicts in the **frontend clone only**
- Never force-push `main` unless the user explicitly requests it

### Step 6: Verify

```powershell
gh repo view sosswced/uploadm8-frontend --web
gh api repos/sosswced/uploadm8-frontend/commits/HEAD --jq '.sha,.commit.message'
```

## Recover source if frontend/ looks empty

If `uploadm8-auth/frontend/` has no HTML after a bad stash:

```powershell
cd C:\Users\Earl\Dev\uploadm8-auth
git stash list
git checkout "stash@{0}^3" -- frontend/
(Get-ChildItem frontend -Filter *.html | Measure-Object).Count
```

## Related skills

- **uploadm8-backend-push** — pushes API/backend to `sosswced/uploadm8-auth` (different repo; do not mix scopes)
- **remove-python-cache-artifacts** — backend only (keyword: **glooo**); not needed for frontend push

## Examples

**User:** `/create-skill push frontend` or "push frontend to GitHub"

1. Pre-flight → robocopy sync → `git status` in frontend clone
2. If user said "push": commit + `git push origin main`

**User:** "Sync frontend but don't commit"

1. Steps 1–3 only; show `git diff --stat` in the frontend clone

**User:** "Push backend and frontend"

1. Run **uploadm8-backend-push** for `uploadm8-auth`
2. Run this skill for `uploadm8-frontend` — **two repos, two pushes**, never one combined commit
