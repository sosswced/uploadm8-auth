"""Admission helpers: maximize throughput without Render OOM."""

from services.worker_admission import (
    active_pipeline_stale_minutes,
    process_dispatch_limit,
    scale_out_hint,
)


def test_process_dispatch_zero_when_memory_blocks():
    assert process_dispatch_limit(local_free_slots=1, memory_blocks=True) == 0


def test_process_dispatch_fills_local_slots_when_healthy():
    assert (
        process_dispatch_limit(
            local_free_slots=1,
            memory_blocks=False,
            fleet={
                "alive_count": 2,
                "workers_hard_pressure": 0,
                "workers_memory_warn": 0,
                "process_slots_in_use": 1,
            },
        )
        == 1
    )


def test_process_dispatch_stops_on_fleet_hard_pressure():
    assert (
        process_dispatch_limit(
            local_free_slots=1,
            memory_blocks=False,
            fleet={"workers_hard_pressure": 1, "alive_count": 1},
        )
        == 0
    )


def test_scale_out_hint_when_backlog_and_healthy_full():
    hint = scale_out_hint(
        {
            "alive_count": 1,
            "process_slots_free": 0,
            "workers_memory_warn": 0,
            "workers_hard_pressure": 0,
        },
        pending=20,
    )
    assert hint is not None
    assert "Scale out" in hint
    assert "WORKER_CONCURRENCY=1" in hint


def test_scale_out_hint_none_when_memory_warn():
    assert (
        scale_out_hint(
            {
                "alive_count": 1,
                "process_slots_free": 0,
                "workers_memory_warn": 1,
                "workers_hard_pressure": 0,
            },
            pending=20,
        )
        is None
    )


def test_active_pipeline_stale_minutes_env(monkeypatch):
    monkeypatch.setenv("ACTIVE_PIPELINE_STALE_MINUTES", "120")
    assert active_pipeline_stale_minutes() == 120
