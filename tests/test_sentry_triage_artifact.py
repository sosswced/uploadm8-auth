"""Unit tests for sentry_triage ship-gate attestation rules."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from scripts.agent.sentry_triage import template, validate_triage


def _cleared_record(**overrides):
    now = datetime.now(timezone.utc).isoformat()
    base = {
        **template(),
        "unresolved_count": 0,
        "actionable_count": 0,
        "cleared": True,
        "mcp_verified": True,
        "mcp_verified_at": now,
        "recorded_at": now,
        "fixed_issue_ids": ["UPLOADM8-1"],
        "waived_issue_ids": [],
    }
    base.update(overrides)
    return base


def test_validate_requires_mcp_verified():
    data = _cleared_record(mcp_verified=False, mcp_verified_at=None)
    ok, blockers = validate_triage(data)
    assert ok is False
    assert any("mcp_verified" in b for b in blockers)


def test_validate_accepts_fresh_mcp_verified():
    ok, blockers = validate_triage(_cleared_record())
    assert ok is True, blockers


def test_validate_rejects_stale_artifact():
    old = (datetime.now(timezone.utc) - timedelta(hours=13)).isoformat()
    data = _cleared_record(mcp_verified_at=old, recorded_at=old)
    ok, blockers = validate_triage(data, max_age_hours=12)
    assert ok is False
    assert any("stale" in b for b in blockers)


def test_validate_rejects_waived_ids():
    data = _cleared_record(waived_issue_ids=["UPLOADM8-X"])
    ok, blockers = validate_triage(data)
    assert ok is False
    assert any("waived" in b for b in blockers)


def test_validate_rejects_actionable_remaining():
    data = _cleared_record(actionable_count=2, cleared=False)
    ok, blockers = validate_triage(data)
    assert ok is False
    assert any("actionable" in b for b in blockers)
