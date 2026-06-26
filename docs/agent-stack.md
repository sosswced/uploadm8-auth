# UploadM8 Agent Stack — Iceberg Levels 4–8+



A production agent configuration for [UploadM8](https://app.uploadm8.com): from curated context through self-healing eval loops, multi-repo orchestration, and ultracode.



## The iceberg (what we implemented)



| Level | Name | UploadM8 implementation |

|-------|------|-------------------------|

| **4** | That Guy | `AGENTS.md`, prompts, rules, MCP, push skills |

| **5** | Degenerate | commands, worktrees, 5-window pattern, subagents |

| **6** | Irredeemable | hooks, parallel Task agents, full skill library |

| **7** | Fallen | overnight, orchestrator, headless, routines, `/loop` |

| **8** | Ascended | eval_loop, self_heal, dynamic_workflow, CI agent, custom MCP |

| **8+** | Ultracode | ultrathink, ultracode, agent-built tooling, agents managing agents |



## Setup checklist (5 minutes)



1. Open this repo in **Cursor** (Agents mode).

2. Confirm **Rules** tab shows rules from `.cursor/rules/` (including `agent-orchestration.mdc`).

3. Confirm **Hooks** tab shows hooks from `.cursor/hooks.json` (restart Cursor if empty).

4. Enable MCP: Sentry, HuggingFace (optional), **uploadm8-agent** custom build.

5. Optional: `pip install -r requirements-agent.txt` for custom MCP server.

6. Run `python run_tests.py verify` — green means the eval harness can run.

7. Try: `/fix-tests`, `/ultrathink`, or `python scripts/agent/dynamic_workflow.py --json`.



## Level 4 — That Guy



**Philosophy:** Context is everything. The model is already capable; your job is to give it the right map.



### Files



- `AGENTS.md` — architecture contracts, escalation graph, full index

- `.cursor/prompts/context-is-everything.md` — core system prompt fragment

- `.cursor/prompts/ascended-operator.md` — Level 8 operator prompt

- `.cursor/rules/uploadm8-core.mdc` — always-on repo conventions

- `.cursor/rules/backend-python.mdc` — routers, services, stages

- `.cursor/rules/frontend-static.mdc` — HTML/JS patterns

- `.cursor/rules/tests-quality.mdc` — pytest and E2E conventions

- `.cursor/rules/agent-orchestration.mdc` — subagents, parallel lanes, workflows

- `.cursor/mcp.json` — Sentry + uploadm8-agent custom MCP



### Skills (project)



| Skill | When |

|-------|------|

| `uploadm8-backend-push` | Push api/core/routers/services to uploadm8-auth |

| `uploadm8-frontend-push` | Sync frontend/ to uploadm8-frontend repo |



## Level 5 — Degenerate



**Philosophy:** One agent, one lane. Parallelize with worktrees and subagents.



### Slash commands



| Command | Purpose |

|---------|---------|

| `/ship-backend` | Scoped backend commit + push workflow |

| `/ship-frontend` | Frontend repo sync |

| `/fix-tests` | Eval loop until tests pass |

| `/overnight` | Run overnight E2E marker suite |

| `/parallel-audit` | Launch explore + bugbot in parallel |

| `/ascended-loop` | Full L8 plan → implement → eval → ship |

| `/five-windows` | Spawn all 5 worktree lanes at once |

| `/ultrathink` | Deep plan before any code |

| `/ultracode` | Eval-as-reward feature loop |

| `/orchestrate` | Agent team with dynamic routing |

| `/multi-repo-ship` | Backend + frontend + CI |

| `/headless-eval` | Background self-heal driver |

| `/loop` | Recurring in-session agent prompt |



### Worktrees



```powershell

.\scripts\agent\worktree_spawn.ps1 -Branch feat/upload-fix -Lane implement

.\scripts\agent\worktree_spawn_all.ps1 -BranchPrefix feat/upload-fix

```



Creates `../uploadm8-auth-wt-<lane>/` — open each in a separate Cursor window.



### Subagent cheat sheet



| Type | Use for |

|------|---------|

| `explore` | Find code, map APIs, grep-heavy discovery |

| `shell` | Git, gh, pytest, deploy scripts |

| `bugbot` | PR / diff review |

| `generalPurpose` | Multi-step implementation |

| `ci-investigator` | Single failing CI check |



## Level 6 — Irredeemable



**Philosophy:** Hooks enforce policy; skills encode expertise; parallel agents compress time.



### Hooks (`.cursor/hooks.json`)



| Event | Behavior |

|-------|----------|

| `sessionStart` | Remind agent to read `AGENTS.md` + slash command index |

| `afterFileEdit` | Suggest `run_tests.py` target for edited paths |

| `beforeShellExecution` | Block `git push --force` to main; warn on `.env` in git add |

| `subagentStop` | Offer eval-loop follow-up after review subagents |



### Parallel pattern



In one agent turn, launch multiple Task subagents:



```

explore: "map upload complete flow"

bugbot: "review uncommitted changes"

```



## Level 7 — Fallen



**Philosophy:** Agents work while you sleep.



### Overnight



```powershell

python run_tests.py overnight

```



Tests marked `@pytest.mark.overnight` in `tests/e2e/`.



### Orchestrator skill



`uploadm8-agent-orchestrator` — structured multi-phase runs:



1. **Plan** — read-only explore subagent

2. **Implement** — generalPurpose with scoped diff

3. **Eval** — `scripts/agent/eval_loop.py`

4. **Review** — bugbot on branch diff

5. **Ship** — push skills (only on user request)



### Background agents



Task tool with `run_in_background: true` for long E2E or explore jobs. See `uploadm8-headless` skill and `/headless-eval`.



### Routines



- `.cursor/automations/README.md` — Cursor Automation templates

- `/loop` command — in-session recurring prompts

- Nightly: `python run_tests.py overnight`

- Push to main: `python scripts/agent/eval_loop.py --mode full --json`



## Level 8 — Ascended



**Philosophy:** Evals drive the loop. Agents fix their own tests. CI is another agent input.



### Agent tooling (`scripts/agent/`)



| Script | Purpose |

|--------|---------|

| `eval_loop.py` | Structured test eval JSON |

| `self_heal.py` | Multi-iteration eval driver |

| `dynamic_workflow.py` | Subagent graph from git diff |

| `ci_status.py` | gh pr checks as JSON |

| `multi_repo_status.py` | Backend + frontend dirty state |

| `register_tool.py` | Agent-built tooling registry |

| `mcp_server.py` | Custom MCP exposing all tools |

| `worktree_spawn.ps1` | Single lane worktree |

| `worktree_spawn_all.ps1` | 5-window spawn |

| `tool_registry.json` | Canonical tool list |



### Eval harness



```powershell

python scripts/agent/eval_loop.py --mode unit --json

python scripts/agent/self_heal.py --mode unit --budget 5 --json

python scripts/agent/dynamic_workflow.py --json

```



### Self-heal loop (`/fix-tests`)



1. Run eval harness

2. If `ok` → stop

3. Parse failures → patch code

4. Re-run eval (max 5 iterations by default)

5. Report diff summary



### Multi-repo orchestration



```powershell

python scripts/agent/multi_repo_status.py --json

```



1. Backend → `uploadm8-backend-push` skill

2. Frontend → `uploadm8-frontend-push` skill

3. `uploadm8-ascended-ci` → `ci_status.py` → fix → push



### Custom MCP (`uploadm8-agent`)



Tools: `eval_run`, `self_heal`, `workflow_plan`, `ci_status`, `multi_repo_status`



```powershell

pip install -r requirements-agent.txt

```



Enable in Cursor Settings → MCP.



## Level 8+ — Ultracode



| Capability | Command / skill |

|------------|-----------------|

| Deep planning | `/ultrathink`, `uploadm8-ultrathink` |

| Eval-as-reward | `/ultracode`, `uploadm8-ultracode` |

| Dynamic routing | `dynamic_workflow.py`, `uploadm8-dynamic-workflows` |

| Agents managing agents | Manager spawns workers per workflow JSON |

| Agent-built tooling | `register_tool.py`, `uploadm8-agent-tooling` |



## Beyond Level 8



| Capability | How |

|------------|-----|

| Production error loop | Sentry MCP → issue → repro test → fix → deploy |

| ML retrain loop | `scripts/ml_engine_run.py` + admin ML observability |

| Cross-org orchestration | Manager agent spawns worker agents per repo |

| CI/CD run by agents | `uploadm8-ascended-ci` + Cursor Automations |



## File tree



```

.cursor/

├── hooks.json

├── hooks/

│   ├── session_start.py

│   ├── after_edit.py

│   ├── before_shell.py

│   └── subagent_stop.py

├── prompts/

│   ├── context-is-everything.md

│   └── ascended-operator.md

├── automations/

│   └── README.md

├── rules/

│   ├── uploadm8-core.mdc

│   ├── backend-python.mdc

│   ├── frontend-static.mdc

│   ├── tests-quality.mdc

│   └── agent-orchestration.mdc

├── commands/

│   ├── ship-backend.md

│   ├── ship-frontend.md

│   ├── fix-tests.md

│   ├── overnight.md

│   ├── parallel-audit.md

│   ├── ascended-loop.md

│   ├── ultrathink.md

│   ├── ultracode.md

│   ├── orchestrate.md

│   ├── five-windows.md

│   ├── multi-repo-ship.md

│   ├── headless-eval.md

│   └── loop.md

├── skills/

│   ├── uploadm8-backend-push/

│   ├── uploadm8-frontend-push/

│   ├── uploadm8-worktrees/

│   ├── uploadm8-agent-orchestrator/

│   ├── uploadm8-eval-loop/

│   ├── uploadm8-ascended-ci/

│   ├── uploadm8-ultrathink/

│   ├── uploadm8-ultracode/

│   ├── uploadm8-multi-repo/

│   ├── uploadm8-headless/

│   ├── uploadm8-routines/

│   ├── uploadm8-dynamic-workflows/

│   └── uploadm8-agent-tooling/

└── mcp.json

AGENTS.md

docs/agent-stack.md

scripts/agent/

├── eval_loop.py

├── self_heal.py

├── dynamic_workflow.py

├── ci_status.py

├── multi_repo_status.py

├── register_tool.py

├── mcp_server.py

├── tool_registry.json

├── worktree_spawn.ps1

└── worktree_spawn_all.ps1

requirements-agent.txt

```



## Going live



This stack is designed to be **committed and shared**. Teammates clone the repo and inherit the full agent configuration.



**Demo script:**



1. Show `AGENTS.md` architecture map

2. Run `python scripts/agent/dynamic_workflow.py --json`

3. Run `/parallel-audit` on a small change

4. Run `python scripts/agent/eval_loop.py --mode unit --json`

5. Show Hooks tab blocking a force-push

6. Optional: enable `uploadm8-agent` MCP and call `eval_run`



That is Levels 4–8+ in one repo.

