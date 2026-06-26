# Context is everything

You are an UploadM8 agent. The model is capable; your job is the **right map**.

## Always read first

1. `AGENTS.md` — architecture, escalation graph, Levels 4–8
2. `docs/agent-stack.md` — full iceberg reference
3. `routers/README.md` — before new endpoints

## Repo contracts

| Layer | Path | Rule |
|-------|------|------|
| API wiring | `app.py` | Lifespan + routers only |
| HTTP | `routers/` | Thin → `services/` |
| Logic | `services/`, `core/` | Business + DB |
| Pipeline | `worker.py`, `stages/` | Async upload stages |
| UI | `frontend/` | Static HTML/JS |
| Tests | `tests/`, `run_tests.py` | Unit + E2E |
| Agents | `.cursor/` | Rules, skills, hooks |

## Dual-repo

- Backend → `sosswced/uploadm8-auth`
- Frontend → `sosswced/uploadm8-frontend`
- Never mix scopes in one push

## Safety

- Never commit `.env` or secrets
- Never force-push main
- Commit/push only when user explicitly asks
- After backend edits: `python run_tests.py unit`
- After frontend edits: `python run_tests.py frontend-lint`

## Escalation

```
User prompt → AGENTS.md → skill/command match → parallel subagents → orchestrator → eval loop → ship
```
