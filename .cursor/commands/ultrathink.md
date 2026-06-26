# Ultrathink — deep plan before code

Follow **uploadm8-ultrathink** skill. Do not write production code until Phase 2 brief is delivered and user approves (unless they said "fully autonomous").

## Steps

1. Load context: `AGENTS.md`, `python scripts/agent/dynamic_workflow.py --json`
2. Launch parallel readonly subagents: 2× explore + bugbot
3. Produce architecture brief (problem, touch map, risks, tests, phased plan, subagent graph)
4. Ask: **Proceed with implementation?**

If approved → chain into `/ascended-loop` or `uploadm8-agent-orchestrator`.
