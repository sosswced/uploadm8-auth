"""
Pikzels once-per-setup gate for /TUP.

First TUP run in a cycle may call Pikzels (thumbnails). Later runs in the same
cycle skip Pikzels until the gate is reset after push/commit (/333).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ARTIFACT = ROOT / "tests" / "e2e" / "artifacts" / "tup_pikzels_setup.json"


def artifact_path() -> Path:
    custom = (os.environ.get("E2E_TUP_PIKZELS_GATE") or "").strip()
    return Path(custom) if custom else DEFAULT_ARTIFACT


def load_gate(path: Path | None = None) -> dict[str, Any]:
    empty = {
        "version": 1,
        "pikzels_used": False,
        "pikzels_used_at": None,
        "reset_at": None,
        "notes": "First /TUP run uses Pikzels; later runs skip until reset after /333",
    }
    p = path or artifact_path()
    if not p.is_file():
        return dict(empty)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return dict(empty)
    if not isinstance(data, dict):
        return dict(empty)
    return data


def save_gate(data: dict[str, Any], path: Path | None = None) -> Path:
    p = path or artifact_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return p


def force_pikzels() -> bool:
    return os.environ.get("E2E_FORCE_PIKZELS", "").lower() in ("1", "true", "yes")


def force_skip_pikzels() -> bool:
    return os.environ.get("E2E_SKIP_PIKZELS", "").lower() in ("1", "true", "yes")


def should_use_pikzels(*, path: Path | None = None) -> bool:
    """True on the first run of a setup cycle (or when E2E_FORCE_PIKZELS=1)."""
    if force_skip_pikzels():
        return False
    if force_pikzels():
        return True
    data = load_gate(path)
    return not bool(data.get("pikzels_used"))


def consume_pikzels_slot(*, path: Path | None = None, note: str = "") -> bool:
    """
    Decide whether this run may use Pikzels. If yes, mark the gate consumed
    so subsequent TUP/heal runs skip until reset_after_ship().
    """
    use = should_use_pikzels(path=path)
    if not use:
        return False
    if force_pikzels() and load_gate(path).get("pikzels_used"):
        # Forced re-run does not rewrite the gate unless first-time.
        return True
    now = datetime.now(timezone.utc).isoformat()
    data = load_gate(path)
    data["pikzels_used"] = True
    data["pikzels_used_at"] = now
    if note:
        data["last_note"] = note[:500]
    save_gate(data, path)
    return True


def reset_after_ship(*, path: Path | None = None, note: str = "reset after push/commit") -> Path:
    """Call after /333 (or explicit --reset-pikzels) so the next cycle uses Pikzels once."""
    now = datetime.now(timezone.utc).isoformat()
    data = {
        "version": 1,
        "pikzels_used": False,
        "pikzels_used_at": None,
        "reset_at": now,
        "notes": note[:500],
    }
    return save_gate(data, path)


def gate_status(*, path: Path | None = None) -> dict[str, Any]:
    data = load_gate(path)
    return {
        "path": str((path or artifact_path()).resolve()),
        "pikzels_used": bool(data.get("pikzels_used")),
        "will_use_pikzels_next": should_use_pikzels(path=path),
        "pikzels_used_at": data.get("pikzels_used_at"),
        "reset_at": data.get("reset_at"),
        "force_pikzels": force_pikzels(),
        "force_skip": force_skip_pikzels(),
    }
