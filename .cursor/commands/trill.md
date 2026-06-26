# TRILL — full stack orchestrator

Run the entire UploadM8 agent stack: route, audit, eval, heal, overnight, multi-repo, CI, optional ship.

## Step 0 — Machine report

```powershell
python scripts/agent/trill.py --mode pre-ship --json
```

Parse `actions[]` and execute each phase below.

## TRILL pipeline

Follow **trill** skill. Track:

```
TRILL:
- [ ] Bootstrap (trill.py JSON green enough to proceed)
- [ ] /parallel-audit
- [ ] /fix-tests if eval red
- [ ] /overnight or /headless-eval if E2E/frontend in diff
- [ ] /multi-repo-ship prep (status only unless user said ship)
- [ ] ci_status + ascended-ci if PR exists
- [ ] Ship only if user said push/ship/go live
```

## Mode selection

| User intent | trill.py flag |
|-------------|---------------|
| General health check | `--mode audit` |
| Before push | `--mode pre-ship` |
| Fix all test failures | `--mode heal` |
| End of day | `--mode overnight` then `/overnight` |

## MCP

Use `trill_run` on uploadm8-agent MCP if shell parsing should be avoided.

## Rules

- Never commit or push without explicit user request in this session.
- Stop and report if trill `actions` include heal but budget exhausted.
- Critical bugbot findings block ship.
