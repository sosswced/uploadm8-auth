# Five windows — parallel agent lanes

Spawn all 5 worktrees for degenerate multi-agent velocity.

```powershell
.\scripts\agent\worktree_spawn_all.ps1 -BranchPrefix feat/my-feature
```

## Lanes

| Window | Lane | Agent focus |
|--------|------|-------------|
| 1 | implement | generalPurpose — feature code |
| 2 | fix | `/fix-tests` eval loop |
| 3 | frontend | frontend-static rules |
| 4 | tests | E2E + eval harness |
| 5 | docs | AGENTS.md, agent stack |

Open each path in a **separate Cursor window**. One agent per lane.

See **uploadm8-worktrees** skill for cleanup.
