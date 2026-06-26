# TRILL — nightly unattended orchestrator

Copy into **Cursor Settings → Automations**.

## TRILL nightly (recommended)

**Trigger:** Schedule — daily 1:00 AM  
**Prompt:**

```
You are TRILL — UploadM8 master orchestrator. Follow the trill skill.

1. Run: python scripts/agent/trill.py --mode audit --json
2. If evals red in report, run /fix-tests logic (max 3 iterations). Do not commit unless this automation config explicitly allows it.
3. Launch /parallel-audit if there are uncommitted changes.
4. If workflow suggests E2E, run in background: python run_tests.py overnight
5. Summarize: trill ok status, actions completed, remaining blockers.

Never force-push main. Never commit .env.
```

## TRILL pre-push gate

**Trigger:** Manual or before deploy  
**Prompt:**

```
Run TRILL pre-ship: python scripts/agent/trill.py --mode pre-ship --json
Then /parallel-audit. Report critical issues only.
Do not ship unless I explicitly said go live in this automation.
```

## TRILL weekly full

**Trigger:** Schedule — Sunday 3:00 AM  
**Prompt:**

```
TRILL full stack health:
1. python scripts/agent/trill.py --mode full --json
2. python scripts/agent/multi_repo_status.py --json
3. Report stack file presence, eval status, repo dirty state, CI status.
Suggest /trill heal if any eval red.
```

## In-session TRILL loop

```
/trill
/loop 1h python scripts/agent/trill.py --mode audit --json and summarize actions
```

See **trill** skill and `.cursor/skills/trill/paths-live.md` for go-live git paths.
