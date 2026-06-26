# UploadM8 frontend push — paths and exclusions

## Source (auth monorepo)

```
C:\Users\Earl\Dev\uploadm8-auth\frontend\
```

Typical contents to sync:

```
*.html
*.js
*.css
_redirects
.gitignore
js/
css/
images/
billing/
compare/
favicon.ico
```

## Destination clone (frontend repo root)

```
C:\Users\Earl\Dev\uploadm8-frontend\
```

Remote:

```
https://github.com/sosswced/uploadm8-frontend
```

## Never sync or commit

```
.env
.env.*
*.pem
*-credentials.json
credentials.json
node_modules/
.git/
desktop.ini
*.log
```

## Robocopy one-liner (PowerShell)

```powershell
robocopy "C:\Users\Earl\Dev\uploadm8-auth\frontend" "C:\Users\Earl\Dev\uploadm8-frontend" /E /XD node_modules .git /XF .env .env.* *.pem *-credentials.json credentials.json desktop.ini
```

## Anti-patterns (learned from prior incidents)

| Do not | Why |
|--------|-----|
| `git stash push -u` on auth repo before frontend work | Stashes untracked `frontend/` and empties the source folder |
| `git add frontend/` in auth repo for a frontend-repo push | Wrong remote; pollutes `uploadm8-auth` |
| Push `node_modules/` | Huge, reproducible with `npm install` |
| `git push --force` to `main` | Unless user explicitly requests |
