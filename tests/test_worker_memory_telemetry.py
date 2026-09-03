"""Worker memory telemetry + 512MB Starter admit defaults."""

from unittest.mock import patch

from core.process_stats import (
    memory_admit_pct,
    memory_hard_pct,
    observability_sample,
)


def test_small_plan_admit_defaults_tighter(monkeypatch):
    monkeypatch.delenv("MEMORY_ADMIT_PCT", raising=False)
    monkeypatch.delenv("MEMORY_HARD_PCT", raising=False)
    monkeypatch.setenv("RENDER_MEMORY_LIMIT_MB", "512")
    assert memory_admit_pct() == 55.0
    assert memory_hard_pct() == 70.0
    assert memory_hard_pct() >= memory_admit_pct()


def test_standard_plan_admit_defaults(monkeypatch):
    monkeypatch.delenv("MEMORY_ADMIT_PCT", raising=False)
    monkeypatch.delenv("MEMORY_HARD_PCT", raising=False)
    monkeypatch.setenv("RENDER_MEMORY_LIMIT_MB", "2048")
    assert memory_admit_pct() == 75.0
    assert memory_hard_pct() == 88.0


def test_observability_sample_includes_children_and_thresholds(monkeypatch):
    monkeypatch.setenv("RENDER_MEMORY_LIMIT_MB", "512")
    monkeypatch.delenv("MEMORY_ADMIT_PCT", raising=False)
    with patch(
        "core.process_stats.sample_memory_mb",
        return_value={
            "rss_mb": 200.0,
            "children_rss_mb": 180.0,
            "effective_rss_mb": 380.0,
            "vms_mb": 500.0,
            "peak_rss_mb": 400.0,
            "limit_mb": 512.0,
            "pct_of_limit": 74.2,
            "small_plan": True,
        },
    ), patch(
        "core.process_stats.sample_load_avg",
        return_value={"load_1m": 1.2, "load_5m": 1.0, "load_15m": 0.8},
    ):
        obs = observability_sample()
        assert obs["children_rss_mb"] == 180.0
        assert obs["effective_rss_mb"] == 380.0
        assert obs["admit_pct"] == 55.0
        assert obs["hard_pct"] == 70.0
        assert obs["memory_pressure"] == "hard"
        assert obs["admission_blocked"] is True


def test_record_and_fetch_memory_samples():
    import asyncio

    from services.worker_memory_telemetry import (
        fetch_global_last_oom_context,
        fetch_last_context,
        fetch_memory_samples,
        record_memory_sample,
        starter_plan_recommendations,
    )

    store = {"lists": {}, "kv": {}}

    class FakePipe:
        def __init__(self):
            self.ops = []

        def lpush(self, key, val):
            self.ops.append(("lpush", key, val))
            return self

        def ltrim(self, key, start, end):
            self.ops.append(("ltrim", key, start, end))
            return self

        def expire(self, key, ttl):
            self.ops.append(("expire", key, ttl))
            return self

        async def execute(self):
            for op in self.ops:
                if op[0] == "lpush":
                    store["lists"].setdefault(op[1], []).insert(0, op[2])
                elif op[0] == "ltrim":
                    key, start, end = op[1], op[2], op[3]
                    store["lists"][key] = store["lists"].get(key, [])[start : end + 1]

    class FakeRedis:
        def pipeline(self, transaction=False):
            return FakePipe()

        async def set(self, key, val, ex=None):
            store["kv"][key] = val

        async def get(self, key):
            return store["kv"].get(key)

        async def lrange(self, key, start, end):
            items = store["lists"].get(key, [])
            if end < 0:
                return items[start:]
            return items[start : end + 1]

    async def _run():
        r = FakeRedis()
        await record_memory_sample(
            r,
            worker_id="srv-srbll",
            sample={
                "rss_mb": 300,
                "children_rss_mb": 150,
                "effective_rss_mb": 450,
                "peak_rss_mb": 450,
                "limit_mb": 512,
                "pct_of_limit": 87.9,
                "memory_pressure": "hard",
                "admission_blocked": True,
                "small_plan": True,
                "load_1m": 2.0,
                "admit_pct": 55,
                "hard_pct": 70,
            },
            jobs={
                "active_process_jobs": [{"upload_id": "u1", "stage": "transcode"}],
                "active_publish_jobs": [],
            },
        )
        hist = await fetch_memory_samples(r, "srv-srbll", limit=10)
        assert len(hist) == 1
        assert hist[0]["children_rss_mb"] == 150
        assert hist[0]["active_process_jobs"][0]["stage"] == "transcode"
        ctx = await fetch_last_context(r, "srv-srbll")
        assert ctx and ctx["pct_of_limit"] == 87.9
        assert "worker:mem:last_oom_context" in store["kv"]
        oom = await fetch_global_last_oom_context(r)
        assert oom and oom.get("likely_oom_precursor") is True
        tips = starter_plan_recommendations(512)
        assert any("MULTIMODAL_PARALLEL" in t for t in tips)
        assert any("cgroup" in t for t in tips)

    asyncio.run(_run())


def test_starter_recommendations_empty_on_standard():
    from services.worker_memory_telemetry import starter_plan_recommendations

    assert starter_plan_recommendations(2048) == []


def test_dangerous_concurrency_flags_starter_multimodal(monkeypatch):
    from services.worker_fleet_watchdog import dangerous_concurrency_warnings

    monkeypatch.setenv("UPLOADM8_PROCESS", "worker")
    monkeypatch.setenv("RENDER", "true")
    monkeypatch.setenv("RENDER_MEMORY_LIMIT_MB", "512")
    monkeypatch.setenv("WORKER_CONCURRENCY", "1")
    monkeypatch.setenv("PUBLISH_CONCURRENCY", "1")
    monkeypatch.setenv("MULTIMODAL_PARALLEL", "true")
    monkeypatch.setenv("VIDEO_INTELLIGENCE_MAX_BYTES", str(100 * 1024 * 1024))
    warns = dangerous_concurrency_warnings()
    assert any("MULTIMODAL_PARALLEL" in w for w in warns)
    assert any("VIDEO_INTELLIGENCE_MAX_BYTES" in w for w in warns)
    assert any("Settings → Video Analyzer" in w for w in warns)


def test_dangerous_concurrency_api_ignores_vi_even_on_small_ram(monkeypatch):
    """Fleet watchdog on Starter API must not alert about worker VI bytes."""
    from services.worker_fleet_watchdog import dangerous_concurrency_warnings

    monkeypatch.setenv("UPLOADM8_PROCESS", "api")
    monkeypatch.setenv("RENDER", "true")
    monkeypatch.setenv("RENDER_MEMORY_LIMIT_MB", "512")
    monkeypatch.setenv("WORKER_CONCURRENCY", "1")
    monkeypatch.setenv("PUBLISH_CONCURRENCY", "1")
    monkeypatch.setenv("MULTIMODAL_PARALLEL", "true")
    monkeypatch.setenv("VIDEO_INTELLIGENCE_MAX_BYTES", str(100 * 1024 * 1024))
    warns = dangerous_concurrency_warnings()
    assert not any("VIDEO_INTELLIGENCE_MAX_BYTES" in w for w in warns)
    assert not any("MULTIMODAL_PARALLEL" in w for w in warns)


def test_dangerous_concurrency_standard_allows_100mib_vi(monkeypatch):
    from services.worker_fleet_watchdog import dangerous_concurrency_warnings

    monkeypatch.setenv("UPLOADM8_PROCESS", "worker")
    monkeypatch.setenv("RENDER", "true")
    monkeypatch.setenv("RENDER_MEMORY_LIMIT_MB", "2048")
    monkeypatch.setenv("WORKER_CONCURRENCY", "1")
    monkeypatch.setenv("PUBLISH_CONCURRENCY", "1")
    monkeypatch.setenv("VIDEO_INTELLIGENCE_MAX_BYTES", str(100 * 1024 * 1024))
    monkeypatch.delenv("MULTIMODAL_PARALLEL", raising=False)
    warns = dangerous_concurrency_warnings()
    assert not any("VIDEO_INTELLIGENCE_MAX_BYTES" in w for w in warns)
