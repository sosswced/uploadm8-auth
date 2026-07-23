"""Memory admission helpers for Render worker OOM protection."""

from unittest.mock import patch

from core.process_stats import (
    blocks_new_process_job,
    memory_hard_pct,
    memory_pressure_level,
)


def test_memory_pressure_levels_from_pct():
    with patch("core.process_stats.memory_admit_pct", return_value=75.0), patch(
        "core.process_stats.memory_hard_pct", return_value=88.0
    ):
        assert memory_pressure_level({"pct_of_limit": 40.0}) == "ok"
        assert memory_pressure_level({"pct_of_limit": 80.0}) == "soft"
        assert memory_pressure_level({"pct_of_limit": 90.0}) == "hard"
        assert memory_pressure_level({"pct_of_limit": None}) == "ok"


def test_blocks_new_process_job_on_soft_or_hard():
    with patch("core.process_stats.memory_admit_pct", return_value=75.0), patch(
        "core.process_stats.memory_hard_pct", return_value=88.0
    ):
        assert blocks_new_process_job({"pct_of_limit": 50.0}) is False
        assert blocks_new_process_job({"pct_of_limit": 76.0}) is True
        assert blocks_new_process_job({"pct_of_limit": 95.0}) is True


def test_memory_hard_pct_above_admit_default():
    from core.process_stats import memory_admit_pct

    assert memory_hard_pct() >= memory_admit_pct()


def test_observability_sample_includes_pressure():
    from core.process_stats import observability_sample

    with patch("core.process_stats.sample_memory_mb", return_value={
        "rss_mb": 100.0,
        "vms_mb": 200.0,
        "peak_rss_mb": 120.0,
        "limit_mb": 2048.0,
        "pct_of_limit": 40.0,
    }), patch("core.process_stats.sample_load_avg", return_value={
        "load_1m": 0.5,
        "load_5m": 0.4,
        "load_15m": 0.3,
    }), patch("core.process_stats.memory_admit_pct", return_value=75.0), patch(
        "core.process_stats.memory_hard_pct", return_value=88.0
    ):
        obs = observability_sample()
        assert obs["memory_pressure"] == "ok"
        assert obs["admission_blocked"] is False
        assert obs["load_1m"] == 0.5
        assert obs["pct_of_limit"] == 40.0


def test_fleet_summary_includes_pressure_fields():
    from services.worker_fleet_snapshot import summarize_fleet

    s = summarize_fleet(
        [
            {
                "status": "alive",
                "worker_concurrency": 1,
                "publish_concurrency": 1,
                "process_slots_in_use": 1,
                "publish_slots_in_use": 0,
                "memory_rss_mb": 1900,
                "memory_limit_mb": 2048,
                "memory_pct": 93,
                "memory_pressure": "hard",
                "admission_blocked": True,
                "load_1m": 2.5,
            },
            {
                "status": "dead",
                "seconds_since_last_beat": 500,
                "worker_concurrency": 1,
            },
            {
                "status": "dead",
                "seconds_since_last_beat": 86400,
                "worker_concurrency": 1,
            },
        ]
    )
    assert s["workers_memory_warn"] == 1
    assert s["workers_hard_pressure"] == 1
    assert s["workers_admission_blocked"] == 1
    assert s["max_load_1m"] == 2.5
    assert s["dead_count"] == 2
    assert s["recent_dead_count"] == 1
