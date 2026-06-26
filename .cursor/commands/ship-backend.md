# Ship backend to uploadm8-auth

Follow the **uploadm8-backend-push** skill end to end.

1. Pre-flight: `git status`, `git remote -v`, confirm no secrets in diff.
2. Run **glooo** (remove-python-cache-artifacts skill) if `__pycache__` present.
3. Stage scoped backend paths only — never `git add .`.
4. Show `git diff --cached --stat` to the user.
5. Commit only if user explicitly asked; draft a why-focused message.
6. `git push -u origin HEAD`
7. Verify with `gh api repos/sosswced/uploadm8-auth/commits/HEAD --jq '.sha,.commit.message'`
