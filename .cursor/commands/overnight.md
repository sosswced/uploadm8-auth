# Overnight test run (Level 7)

Unattended E2E suite for background or end-of-day runs.

1. Confirm API is reachable if E2E requires it (check `tests/e2e/conftest.py`).
2. Run: `python run_tests.py overnight`
3. If running in background, use Task subagent with `run_in_background: true` and `subagent_type: shell`.
4. On completion, summarize: pass count, failures, flaky tests, suggested fixes.
5. For failures, offer `/fix-tests` or create a focused issue list.

Do not commit fixes unless the user asks.
