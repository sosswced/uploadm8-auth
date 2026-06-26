# Headless eval — background self-heal driver

Follow **uploadm8-headless** skill. Run eval iterations without blocking.

## Command

Launch background shell subagent (`run_in_background: true`):

```powershell
python scripts/agent/self_heal.py --mode unit --budget 5 --json
```

Or for overnight:

```powershell
python run_tests.py overnight
```

## On completion

1. Parse JSON summary
2. If red → offer `/fix-tests`
3. If green → report success

Parent agent continues other work while shell runs.
