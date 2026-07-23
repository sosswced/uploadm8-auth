"""Unit tests for Render overload defense: fleet watchdog + Render API helpers."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from services.render_platform import (
    build_observability_health,
    classify_event_severity,
    summarize_metric_timeseries,
    summarize_render_events,
    _unwrap_list_payload,
)
from services.worker_fleet_watchdog import (
    dangerous_concurrency_warnings,
    evaluate_fleet_alerts,
    evaluate_render_event_alerts,
)


def test_unwrap_list_payload_nested_event():
    raw = [
        {"cursor": "a", "event": {"id": "1", "type": "server_failed"}},
        {"cursor": "b", "service": {"id": "srv-1", "name": "worker"}},
    ]
    out = _unwrap_list_payload(raw)
    assert out[0]["type"] == "server_failed"
    assert out[0]["_cursor"] == "a"
    assert out[1]["name"] == "worker"


def test_classify_event_severity():
    assert classify_event_severity("server_failed") == "critical"
    assert classify_event_severity("server_restarted") == "critical"
    assert classify_event_severity("autoscaling_started") == "info"
    assert classify_event_severity("deploy_ended") == "info"
    assert classify_event_severity("mystery") == "warning"


def test_summarize_render_events():
    events = [
        {"type": "server_failed", "severity": "critical"},
        {"type": "autoscaling_started", "severity": "info"},
        {"type": "server_restarted", "severity": "critical"},
    ]
    s = summarize_render_events(events)
    assert s["critical_count"] == 2
    assert s["autoscale_count"] == 1
    assert s["has_recent_failure"] is True
    assert s["latest_critical"]["type"] == "server_failed"


def test_summarize_metric_timeseries():
    payload = {
        "timeseries": [
            {
                "values": [
                    {"timestamp": "t1", "value": 0.2},
                    {"timestamp": "t2", "value": 0.9},
                    {"timestamp": "t3", "value": 0.4},
                ]
            }
        ]
    }
    s = summarize_metric_timeseries(payload)
    assert s["latest"] == 0.4
    assert s["max"] == 0.9
    assert s["points"] == 3


def test_build_observability_health_critical_on_failures():
    health = build_observability_health(
        events_summary={"has_recent_failure": True, "critical_count": 2},
        instances={"ok": True, "running_count": 0, "instance_count": 1},
        metrics={"cpu": {"max": 0.95}, "memory": {"latest": None}},
        deploys={"latest": {"status": "live"}},
        oom_logs={"count": 2},
    )
    assert health["status"] == "critical"
    assert "recent_render_failure" in health["flags"]
    assert "oom_log_hits" in health["flags"]


def test_build_observability_health_healthy():
    health = build_observability_health(
        events_summary={"has_recent_failure": False},
        instances={"ok": True, "running_count": 2, "instance_count": 2},
        metrics={"cpu": {"max": 0.2}, "memory": {"latest": 100}},
        deploys={"latest": {"status": "live"}},
        oom_logs={"count": 0},
    )
    assert health["status"] == "healthy"
    assert health["score"] >= 80


def test_evaluate_fleet_down():
    alerts = evaluate_fleet_alerts(
        {
            "worker_count": 2,
            "alive_count": 0,
            "stale_count": 0,
            "dead_count": 2,
            "recent_dead_count": 2,
            "workers_memory_warn": 0,
        },
        uploads={"processing": 3},
        queues={"total_pending": 4},
    )
    types = [a.incident_type for a in alerts]
    assert "worker_fleet_down" in types
    assert alerts[0].severity == "critical"


def test_evaluate_ignores_historical_dead_when_alive():
    alerts = evaluate_fleet_alerts(
        {
            "worker_count": 103,
            "alive_count": 2,
            "stale_count": 0,
            "dead_count": 101,
            "recent_dead_count": 0,
            "workers_memory_warn": 0,
            "process_capacity": 2,
            "process_slots_free": 2,
            "process_slots_in_use": 0,
        },
        uploads={"processing": 0},
        queues={"total_pending": 0},
    )
    assert all(a.incident_type != "worker_instance_dead" for a in alerts)


def test_evaluate_memory_and_overload():
    alerts = evaluate_fleet_alerts(
        {
            "worker_count": 1,
            "alive_count": 1,
            "stale_count": 0,
            "dead_count": 0,
            "workers_memory_warn": 1,
            "process_capacity": 1,
            "process_slots_free": 0,
            "process_slots_in_use": 1,
        },
        uploads={"processing": 2},
        queues={"total_pending": 12},
        queue_pending_threshold=8,
    )
    types = {a.incident_type for a in alerts}
    assert "worker_memory_pressure" in types
    assert "worker_overload" in types


def test_evaluate_no_alerts_when_healthy():
    alerts = evaluate_fleet_alerts(
        {
            "worker_count": 1,
            "alive_count": 1,
            "stale_count": 0,
            "dead_count": 0,
            "workers_memory_warn": 0,
            "process_capacity": 1,
            "process_slots_free": 1,
            "process_slots_in_use": 0,
        },
        uploads={"processing": 0},
        queues={"total_pending": 0},
    )
    assert alerts == []


def test_evaluate_render_event_alerts_dedupes_type():
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    alerts = evaluate_render_event_alerts(
        [
            {"type": "server_failed", "severity": "critical", "timestamp": now},
            {"type": "server_failed", "severity": "critical", "timestamp": now},
            {"type": "autoscaling_started", "severity": "info", "timestamp": now},
        ]
    )
    assert len(alerts) == 1
    assert alerts[0].incident_type == "render_event_server_failed"


def test_evaluate_render_event_alerts_ignores_stale():
    alerts = evaluate_render_event_alerts(
        [
            {
                "type": "server_failed",
                "severity": "critical",
                "timestamp": "2026-07-22T12:34:56.422235Z",
            }
        ],
        max_age_sec=1200,
    )
    assert alerts == []


def test_dangerous_concurrency_warnings(monkeypatch):
    monkeypatch.setenv("RENDER", "true")
    monkeypatch.setenv("WORKER_CONCURRENCY", "2")
    monkeypatch.setenv("PUBLISH_CONCURRENCY", "5")
    monkeypatch.setenv("RENDER_MEMORY_LIMIT_MB", "2048")
    monkeypatch.setenv("WORKER_LANE", "full")
    warns = dangerous_concurrency_warnings()
    assert any("WORKER_CONCURRENCY=2" in w for w in warns)
    assert any("PUBLISH_CONCURRENCY=5" in w for w in warns)


def test_dangerous_concurrency_ok_at_one(monkeypatch):
    monkeypatch.setenv("RENDER", "true")
    monkeypatch.setenv("WORKER_CONCURRENCY", "1")
    monkeypatch.setenv("PUBLISH_CONCURRENCY", "1")
    monkeypatch.setenv("RENDER_MEMORY_LIMIT_MB", "2048")
    assert dangerous_concurrency_warnings() == []


def test_build_render_live_snapshot_unconfigured(monkeypatch):
    for key in (
        "RENDER_MONITOR_API_KEY",
        "RENDER_MONITOR_SERVICE_ID",
        "RENDER_MONITOR_OWNER_ID",
        "RENDER_API_KEY",
        "RENDER_WORKER_SERVICE_ID",
        "RENDER_OWNER_ID",
        "RENDER_SERVICE_ID",
    ):
        monkeypatch.delenv(key, raising=False)
    from services.render_platform import build_render_live_snapshot

    snap = asyncio.run(build_render_live_snapshot())
    assert snap["configured"] is False
    assert snap["ok"] is False
    assert "RENDER_MONITOR_API_KEY" in snap["message"]


def test_render_monitor_canonical_env_preferred(monkeypatch):
    monkeypatch.setenv("RENDER_MONITOR_API_KEY", "rnd_new")
    monkeypatch.setenv("RENDER_MONITOR_SERVICE_ID", "srv_new")
    monkeypatch.setenv("RENDER_API_KEY", "rnd_old")
    monkeypatch.setenv("RENDER_WORKER_SERVICE_ID", "srv_old")
    monkeypatch.setenv("RENDER_SERVICE_ID", "srv_api_auto_ignore")
    from services.render_platform import render_api_key, render_worker_service_id

    assert render_api_key() == "rnd_new"
    assert render_worker_service_id() == "srv_new"


def test_render_monitor_legacy_fallback(monkeypatch):
    for key in (
        "RENDER_MONITOR_API_KEY",
        "RENDER_MONITOR_SERVICE_ID",
        "RENDER_SERVICE_ID",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("RENDER_API_KEY", "rnd_legacy")
    monkeypatch.setenv("RENDER_WORKER_SERVICE_ID", "srv_legacy")
    from services.render_platform import render_api_key, render_worker_service_id

    assert render_api_key() == "rnd_legacy"
    assert render_worker_service_id() == "srv_legacy"


def test_run_fleet_watchdog_once_records_incident(monkeypatch):
    monkeypatch.setenv("WATCHDOG_RENDER_API_ENABLED", "0")
    monkeypatch.delenv("RENDER", raising=False)

    fleet_snap = {
        "fleet": {
            "worker_count": 1,
            "alive_count": 0,
            "stale_count": 0,
            "dead_count": 1,
            "workers_memory_warn": 0,
            "process_capacity": 0,
            "process_slots_free": 0,
            "process_slots_in_use": 0,
        },
        "uploads": {"processing": 1},
        "redis_queues": {"total_pending": 2},
        "timestamp": "now",
    }

    with patch(
        "services.worker_fleet_snapshot.build_worker_fleet_snapshot",
        new=AsyncMock(return_value=fleet_snap),
    ), patch(
        "services.ops_incidents.record_operational_incident",
        new=AsyncMock(return_value="inc-1"),
    ) as rec:
        from services.worker_fleet_watchdog import run_fleet_watchdog_once

        result = asyncio.run(run_fleet_watchdog_once(MagicMock(), None))
        assert result["alert_count"] >= 1
        assert "inc-1" in result["incident_ids"]
        assert rec.await_count >= 1
        call_kw = rec.await_args.kwargs
        assert call_kw["source"] == "worker_fleet_watchdog"
        assert call_kw["incident_type"] == "worker_fleet_down"
