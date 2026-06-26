# UploadM8 Agent Stack



> **Context is everything.** This repo ships a full Cursor agent stack from Level 4 (*That Guy*) through Level 8+ (*Ascended / Ultracode*). Read this file first in every agent session.



Custom prompts: `.cursor/prompts/` · Rules: `.cursor/rules/` · Skills: `.cursor/skills/` · MCP: `.cursor/mcp.json`



## Repo map (read before touching code)



| Layer | Path | Contract |

|-------|------|----------|

| API wiring | `app.py` | Lifespan, middleware, `include_router` only — no business handlers |

| HTTP surface | `routers/*.py` | Thin handlers → `services/` |

| Business logic | `services/`, `core/` | DB, orchestration, shared helpers |

| Worker pipeline | `worker.py`, `stages/` | Transcode, captions, thumbnails, publish |

| Static UI | `frontend/` | HTML + vanilla JS; calls `/api/*` via `js/api-base.js` |

| Tests | `tests/`, `run_tests.py` | Unit, contract, E2E (Playwright) |

| Agent config | `.cursor/` | Rules, skills, commands, hooks, prompts, automations |

| Agent tooling | `scripts/agent/` | Eval harness, MCP server, workflow planner, registry |



**Dual-repo deploy:** backend → `sosswced/uploadm8-auth`, frontend → `sosswced/uploadm8-frontend`. Use the push skills; never mix scopes.



## Level 4 — That Guy



Foundation layer. Always on.



- **This file** (`AGENTS.md`) — agent onboarding and architecture contracts

- **Custom system prompts** — `.cursor/prompts/context-is-everything.md`, `ascended-operator.md`

- **Cursor rules** — `.cursor/rules/*.mdc` (core, backend, frontend, tests, orchestration)

- **MCP servers** — `.cursor/mcp.json` (Sentry, HuggingFace plugins, **uploadm8-agent** custom build)

- **Skills** — `.cursor/skills/*/SKILL.md` (push, eval, orchestrator, ultrathink, …)



### Agent defaults



1. Read `routers/README.md` before adding endpoints.

2. Run `python run_tests.py unit` after backend changes; `frontend-lint` after HTML/JS edits.

3. Never stage `.env`, credentials, or `__pycache__`.

4. Commit only when the user explicitly asks.



## Level 5 — Degenerate



Parallel velocity.



- **Slash commands** — `.cursor/commands/*.md` (`/ship-backend`, `/fix-tests`, `/five-windows`, …)

- **Git worktrees** — `scripts/agent/worktree_spawn.ps1`, `worktree_spawn_all.ps1` + `uploadm8-worktrees` skill

- **Subagents** — spawn `explore`, `shell`, `bugbot`, `generalPurpose`, `ci-investigator`

- **5-window pattern** — one worktree per lane: `implement`, `fix`, `frontend`, `tests`, `docs`



## Level 6 — Irredeemable



Autonomous guardrails.



- **Hooks** — `.cursor/hooks.json` + `.cursor/hooks/*.py`

  - `sessionStart` → inject stack pointer + slash commands

  - `afterFileEdit` → suggest targeted tests

  - `beforeShellExecution` → block secrets / force-push to main

  - `subagentStop` → chain eval loop when review subagents finish

- **Skills** — reusable workflows the agent invokes by description match

- **Parallel agents** — Task tool with multiple `subagent_type` in one turn



## Level 7 — Fallen



Scheduled and unsupervised runs.



- **Overnight suite** — `python run_tests.py overnight` (pytest `-m overnight`)

- **Routines** — `.cursor/automations/README.md` + `/loop` command + Cursor Automations

- **Agent teams** — `uploadm8-agent-orchestrator` skill: planner → implementer → tester → reviewer

- **Headless** — background Task subagents + `/headless-eval` + `uploadm8-headless` skill



## Level 8 — Ascended



Eval-driven, self-healing loops.



- **Eval harness** — `python scripts/agent/eval_loop.py` → JSON report

- **Self-heal driver** — `python scripts/agent/self_heal.py` → multi-iteration JSON

- **Dynamic workflows** — `python scripts/agent/dynamic_workflow.py` → subagent graph from diff

- **Fix-own-tests loop** — `/fix-tests` command: eval → patch → re-run until green

- **Multi-repo** — `multi_repo_status.py` + `uploadm8-multi-repo` skill + `/multi-repo-ship`

- **CI agent** — `ci_status.py` + `uploadm8-ascended-ci` skill: PR checks → fix → push

- **Custom MCP** — `scripts/agent/mcp_server.py` exposes eval, workflow, CI tools



## Level 8+ — Ultracode



Eval as reward signal; architecture PRs backed by green evals.



- **Ultrathink** — `/ultrathink` + `uploadm8-ultrathink` — deep plan before code

- **Ultracode** — `/ultracode` + `uploadm8-ultracode` — iterate until eval green

- **Agent-built tooling** — `scripts/agent/register_tool.py` + `tool_registry.json`



## Quick commands



```powershell

# Verify environment

python run_tests.py verify



# Fast unit gate

python run_tests.py unit



# Frontend inline lint

python run_tests.py frontend-lint



# Eval loop (structured JSON for agents)

python scripts/agent/eval_loop.py --mode unit --json



# Self-heal iterations

python scripts/agent/self_heal.py --mode unit --budget 5 --json



# Dynamic workflow from git diff

python scripts/agent/dynamic_workflow.py --json



# Multi-repo status

python scripts/agent/multi_repo_status.py --json



# CI status (requires gh)

python scripts/agent/ci_status.py --json



# Spawn parallel worktree

.\scripts\agent\worktree_spawn.ps1 -Branch feat/my-change -Lane implement



# Spawn all 5 windows

.\scripts\agent\worktree_spawn_all.ps1 -BranchPrefix feat/my-change



# Custom MCP server (optional)

pip install -r requirements-agent.txt

python scripts/agent/mcp_server.py

```



## Slash command index



| Command | Level | Purpose |

|---------|-------|---------|

| `/fix-tests` | 8 | Self-heal eval loop |

| `/ascended-loop` | 8 | Plan → implement → eval → review |

| `/ultrathink` | 8+ | Deep architecture plan |

| `/ultracode` | 8+ | Eval-driven feature loop |

| `/orchestrate` | 7 | Agent team with dynamic routing |

| `/parallel-audit` | 6 | explore + bugbot in parallel |

| `/five-windows` | 5 | Spawn 5 worktree lanes |

| `/ship-backend` | 5 | Scoped backend push |

| `/ship-frontend` | 5 | Frontend repo sync |

| `/multi-repo-ship` | 8 | Both repos + CI |

| `/headless-eval` | 7 | Background self-heal |

| `/overnight` | 7→8+ | Full pipeline → ship gate → `/333` when green (ultrathink, five-windows, checklist, E2E, ultracode, Sentry, `ship_gate.py`) |

| `/loop` | 7 | Recurring in-session prompt |

| `/trill` | 8+ | Master orchestrator — full stack in one command |



## Skill index



| Skill | Purpose |

|-------|---------|

| `trill` | Master orchestrator — route, audit, eval, heal, CI, ship prep |

| `uploadm8-backend-push` | Backend scoped push |

| `uploadm8-frontend-push` | Frontend repo sync |

| `uploadm8-worktrees` | Parallel git worktrees |

| `uploadm8-eval-loop` | Eval harness + self-heal |

| `uploadm8-agent-orchestrator` | Multi-phase agent team |

| `uploadm8-ascended-ci` | CI/CD agent workflow |

| `uploadm8-ultrathink` | Deep planning |

| `uploadm8-ultracode` | Eval-as-reward development |

| `uploadm8-multi-repo` | Cross-repo orchestration |

| `uploadm8-headless` | Background agents |

| `uploadm8-routines` | Automations + /loop |

| `uploadm8-dynamic-workflows` | Diff-driven agent graph |

| `uploadm8-agent-tooling` | Register agent scripts |



## Escalation graph



```

User prompt

    │

    ▼

AGENTS.md + prompts + rules (L4)

    │

    ├─► dynamic_workflow.py → pick command/skill (L8)

    │

    ├─► slash command or skill match (L5-L6)

    │

    ├─► parallel subagents (L6)

    │       explore │ shell │ bugbot │ ci-investigator

    │

    ├─► hooks gate edits & shell (L6)

    │

    ├─► orchestrator chain (L7)

    │       plan → implement → eval_loop → review

    │

    ├─► headless / overnight / routines (L7)

    │

    └─► ascended self-heal (L8+)

            eval_loop → fix → eval_loop → ci_status → ship

```



## MCP servers



| Server | Source | Tools |

|--------|--------|-------|

| `sentry` | Plugin | Production error triage |

| HuggingFace | Plugin | Models, datasets, jobs |

| `uploadm8-agent` | Custom build | eval_run, self_heal, workflow_plan, ci_status, multi_repo_status |



Enable in Cursor Settings → MCP. Custom server requires `pip install -r requirements-agent.txt`.



## Beyond Level 8



| Capability | How |

|------------|-----|

| Agent-built tooling | `register_tool.py` + extend `mcp_server.py` |

| Cross-repo orchestration | `/multi-repo-ship` + manager/worker subagents |

| Production error loop | Sentry MCP → repro test → fix → deploy |

| ML retrain loop | `scripts/ml_engine_run.py` + admin ML observability |

| Agents managing agents | `dynamic_workflow.py` → spawn workers per lane |



See `docs/agent-stack.md` for the full iceberg reference and setup checklist.

