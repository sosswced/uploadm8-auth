# Ascended loop (Level 8 full cycle)

Full autonomous cycle: plan → implement → eval → review → (optional ship).

## Phases

### 1. Plan (explore, readonly)
Map affected modules. Identify test targets. Output a 3–5 step plan.

### 2. Implement (generalPurpose or direct)
Execute the plan with minimal scope. Match existing code style.

### 3. Eval (mandatory)
```powershell
python scripts/agent/eval_loop.py --mode unit --json
```
If failures, run `/fix-tests` logic (max 3 iterations in this phase).

### 4. Review (bugbot, readonly)
Review the full diff. Address critical findings.

### 5. Ship (only if user said "ship" or "push")
- Backend → `/ship-backend`
- Frontend → `/ship-frontend`

## Rules

- Never commit or push without explicit user request in this session.
- Report structured summary at each phase boundary.
- If eval stays red after budget, stop and ask the user.
