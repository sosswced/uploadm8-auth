# Ultracode — eval as reward signal

Follow **uploadm8-ultracode** skill. Tests are the score; iterate until green.

## Loop

1. Small focused diff (one hypothesis)
2. `python scripts/agent/eval_loop.py --mode unit --json`
3. If `ok: false` → fix root cause, repeat (max 5 iterations)
4. If stalled 3× on same test → launch `explore` subagent
5. Green → summarize score (iterations, failures fixed)
6. Optional: bugbot review

Never delete tests to green. Never skip hooks.

For features spanning backend + frontend, run `dynamic_workflow.py` first.
