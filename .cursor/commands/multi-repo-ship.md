# Multi-repo ship — backend + frontend orchestration

Follow **uploadm8-multi-repo** skill.

## Status

```powershell
python scripts/agent/multi_repo_status.py --json
```

## Flow

1. Confirm which repos are dirty
2. Backend first if API contract changed → `/ship-backend`
3. Frontend sync → `/ship-frontend`
4. Ascended CI on each PR → **uploadm8-ascended-ci** skill

```powershell
python scripts/agent/ci_status.py --json
```

Never commit or push without explicit user request.
