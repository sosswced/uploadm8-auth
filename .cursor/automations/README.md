# Cursor Automations — UploadM8 templates (Level 7)

Copy into **Cursor Settings → Automations**. Adjust triggers to your workflow.

## Nightly overnight E2E

**Trigger:** Schedule — daily 2:00 AM  
**Prompt:**
```
Run python run_tests.py overnight in the uploadm8-auth repo.
Summarize pass/fail counts. If failures, list test nodeids and suggest /fix-tests steps.
Do not commit unless I explicitly asked in this automation config.
```

## Push to main — eval gate

**Trigger:** Git push to `main` on `sosswced/uploadm8-auth`  
**Prompt:**
```
Run python scripts/agent/eval_loop.py --mode full --json.
If ok is false, open a summary of failures and suggested_command.
Do not auto-commit fixes.
```

## PR opened — CI watch

**Trigger:** Pull request opened  
**Prompt:**
```
Run python scripts/agent/ci_status.py --json for the current PR.
If failing_checks is non-empty, launch ci-investigator subagent per check and summarize.
```

## Weekly agent stack health

**Trigger:** Schedule — weekly Monday 9:00 AM  
**Prompt:**
```
Verify UploadM8 agent stack:
1. python run_tests.py verify
2. python scripts/agent/eval_loop.py --mode unit --json
3. python scripts/agent/dynamic_workflow.py --json
Report status. Flag any red evals.
```

## In-session alternative

Use `/loop 1h run eval_loop` instead of Automations for local recurring checks.

See **uploadm8-routines** skill.
