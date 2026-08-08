"""Worker memory telemetry — Redis ring history + last-known OOM debug context.

Gives Admin KPI /ops full visibility into RSS vs stage/job on small Render plans
(512MB Starter) without requiring a page refresh or Render dashboard dig.
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("uploadm8.worker_memory_telemetry")

_SAMPLE_KEY = "worker:mem:samples:{worker_id}"
_LAST_CTX_KEY = "worker:mem:last_context:{worker_id}"
_GLOBAL_LAST_OOM_KEY = "worker:mem:last_oom_context"
_MAX_SAMPLES = max(30, min(int(os.environ.get("WORKER_MEM_SAMPLE_RING") or 120), 500))
_SAMPLE_TTL_SEC = 6 * 3600


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def record_memory_sample(
    redis: Any,
    *,
    worker_id: str,
    sample: Dict[str, Any],
    jobs: Optional[Dict[str, Any]] = None,
    loops: Optional[List[str]] = None,
) -> None:
    """Push one heartbeat/stage sample onto a per-worker Redis ring."""
    if redis is None or not worker_id:
        return
    try:
        payload = {
            "ts": _now_iso(),
            "epoch": time.time(),
            "worker_id": str(worker_id),
            "rss_mb": sample.get("rss_mb"),
            "effective_rss_mb": sample.get("effective_rss_mb"),
            "peak_rss_mb": sample.get("peak_rss_mb"),
            "children_rss_mb": sample.get("children_rss_mb"),
            "limit_mb": sample.get("limit_mb"),
            "pct_of_limit": sample.get("pct_of_limit"),
            "admit_pct": sample.get("admit_pct"),
            "hard_pct": sample.get("hard_pct"),
            "memory_pressure": sample.get("memory_pressure"),
            "admission_blocked": sample.get("admission_blocked"),
            "small_plan": sample.get("small_plan"),
            "load_1m": sample.get("load_1m"),
            "active_process_jobs": (jobs or {}).get("active_process_jobs") or [],
            "active_publish_jobs": (jobs or {}).get("active_publish_jobs") or [],
            "loops": list(loops or []),
        }
        key = _SAMPLE_KEY.format(worker_id=worker_id)
        pipe = redis.pipeline(transaction=False)
        pipe.lpush(key, json.dumps(payload, default=str))
        pipe.ltrim(key, 0, _MAX_SAMPLES - 1)
        pipe.expire(key, _SAMPLE_TTL_SEC)
        await pipe.execute()

        # Keep a sticky "last known" blob so OOM kills still leave a breadcrumb.
        ctx_key = _LAST_CTX_KEY.format(worker_id=worker_id)
        await redis.set(ctx_key, json.dumps(payload, default=str), ex=_SAMPLE_TTL_SEC)

        pressure = str(sample.get("memory_pressure") or "")
        pct = sample.get("pct_of_limit")
        if pressure == "hard" or (isinstance(pct, (int, float)) and pct >= 90):
            await redis.set(
                _GLOBAL_LAST_OOM_KEY,
                json.dumps({**payload, "likely_oom_precursor": True}, default=str),
                ex=24 * 3600,
            )
    except Exception as e:
        logger.debug("record_memory_sample skipped: %s", e)


async def fetch_memory_samples(
    redis: Any,
    worker_id: str,
    *,
    limit: int = 60,
) -> List[Dict[str, Any]]:
    if redis is None or not worker_id:
        return []
    try:
        raw = await redis.lrange(
            _SAMPLE_KEY.format(worker_id=worker_id),
            0,
            max(1, min(int(limit), _MAX_SAMPLES)) - 1,
        )
    except Exception:
        return []
    out: List[Dict[str, Any]] = []
    for item in raw or []:
        try:
            if isinstance(item, bytes):
                item = item.decode("utf-8", errors="replace")
            d = json.loads(item)
            if isinstance(d, dict):
                out.append(d)
        except Exception:
            continue
    return out


async def fetch_last_context(redis: Any, worker_id: str) -> Optional[Dict[str, Any]]:
    if redis is None or not worker_id:
        return None
    try:
        raw = await redis.get(_LAST_CTX_KEY.format(worker_id=worker_id))
        if not raw:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        d = json.loads(raw)
        return d if isinstance(d, dict) else None
    except Exception:
        return None


async def fetch_global_last_oom_context(redis: Any) -> Optional[Dict[str, Any]]:
    if redis is None:
        return None
    try:
        raw = await redis.get(_GLOBAL_LAST_OOM_KEY)
        if not raw:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        d = json.loads(raw)
        return d if isinstance(d, dict) else None
    except Exception:
        return None


def starter_plan_recommendations(limit_mb: Optional[float]) -> List[str]:
    """Actionable env tips when the instance is a 512MB Starter."""
    tips: List[str] = []
    if limit_mb is None or float(limit_mb) > 768:
        return tips
    tips.append("Plan is ≤512–768MB — keep WORKER_CONCURRENCY=1 and PUBLISH_CONCURRENCY=1.")
    tips.append("Set MULTIMODAL_PARALLEL=false so audio/vision/VI do not run together.")
    tips.append(
        "On Starter, keep VIDEO_INTELLIGENCE_MAX_BYTES≤20971520 (or unset — runtime clamps). "
        "Users opt out via Settings → Video Analyzer (VI); never set VIDEO_INTELLIGENCE_STAGE_ENABLED=false."
    )
    tips.append("Set FFMPEG_THREADS=1 and prefer WATERMARK_SINGLE_PASS=true.")
    tips.append(
        "RAM is detected from the container cgroup — RENDER_MEMORY_LIMIT_MB is optional "
        "(only set it if cgroup is wrong)."
    )
    tips.append("Disable non-essential loops on this instance (WORKER_ENABLE_KPI_COLLECTOR=false, etc.).")
    tips.append("Prefer upgrading the worker to ≥2GB Standard — HD encode + AI does not fit 512MB.")
    return tips
