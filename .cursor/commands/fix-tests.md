# Fix tests (Level 8 eval loop)

Self-healing test loop. Max **5** iterations unless user sets a different budget.

## Loop

1. Run `python scripts/agent/eval_loop.py --mode unit --json`
2. If `ok: true` → report success and stop.
3. Parse `failures` array — fix the root cause (not symptoms).
4. Re-run eval harness.
5. Repeat until green or budget exhausted.

## Scope rules

- Minimize diff — fix only what failures require.
- Do not delete tests to make green.
- Do not skip hooks or use `--no-verify`.
- After green, summarize what changed and why.

## Escalation

If 2+ iterations fail on the same test, launch an `explore` subagent to map the failure domain before patching again.
