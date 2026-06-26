# Parallel audit (Level 6)

Launch **two subagents in parallel** in a single turn:

1. **explore** (quick): Map the code areas touched by current uncommitted changes. Return file list + risk areas.
2. **bugbot** (readonly): Review branch changes or uncommitted diff. Return findings by severity.

Synthesize both reports into one actionable summary:

- Critical issues (must fix before ship)
- Suggestions (nice to have)
- Test gaps (which `run_tests.py` mode to run)

Do not implement fixes unless the user asks — this command is audit-only.
