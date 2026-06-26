# TRILL go-live — git paths to push agent stack

Stage these for the agent stack to be live for all teammates cloning the repo.

## Required (agent stack)

```
.cursor/
AGENTS.md
docs/agent-stack.md
requirements-agent.txt
scripts/agent/
run_tests.py
pytest.ini
```

## Backend push scope (also stage when changed)

Per `uploadm8-backend-push` skill:

```
api/ core/ docs/ jobs/ migrations/ routers/ schemas/ scripts/ services/ stages/ tools/
app.py worker.py Dockerfile requirements.txt requirements-api.txt requirements-lock.txt runtime.txt
.gitignore
```

## One-shot stage (PowerShell)

```powershell
git add .cursor/ AGENTS.md docs/agent-stack.md requirements-agent.txt scripts/agent/ run_tests.py pytest.ini
git add api/ core/ docs/ jobs/ migrations/ routers/ schemas/ scripts/ services/ stages/ tools/
git add app.py worker.py .gitignore
# Add other root files if changed: run_tests.py, .env.example, etc.
git diff --cached --stat
```

## After push — activate locally

1. Restart Cursor (reload rules, hooks, commands, skills)
2. Settings → MCP → enable `uploadm8-agent` + Sentry
3. `pip install -r requirements-agent.txt`
4. `python scripts/agent/trill.py --json` — should return TRILL report
5. Try `/trill` in chat

## Automations (optional, Cursor Settings → Automations)

Copy from `.cursor/automations/trill.md` for nightly unattended TRILL.

## Verify live

```powershell
python scripts/agent/trill.py --mode pre-ship --json
```

`ok: true` + no critical bugbot findings → ready for `/multi-repo-ship` when user asks.
