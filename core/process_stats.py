"""Process memory and Render instance metadata for worker/API observability."""
from __future__ import annotations

import os
import platform
import socket
from typing import Any, Dict, Optional

try:
    import resource
except ImportError:
    resource = None  # Windows — use /proc only (unavailable locally)

_peak_rss_mb: float = 0.0


def _read_proc_rss_vms_mb() -> tuple[Optional[float], Optional[float]]:
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            rss_kb = None
            vms_kb = None
            for line in f:
                if line.startswith("VmRSS:"):
                    rss_kb = int(line.split()[1])
                elif line.startswith("VmSize:"):
                    vms_kb = int(line.split()[1])
            if rss_kb is not None:
                return rss_kb / 1024.0, (vms_kb / 1024.0) if vms_kb else None
    except (OSError, ValueError, IndexError):
        pass
    return None, None


def memory_limit_mb() -> Optional[float]:
    for key in ("RENDER_MEMORY_LIMIT_MB", "MEMORY_LIMIT_MB", "WEB_MEMORY_LIMIT_MB"):
        raw = (os.environ.get(key) or "").strip()
        if raw:
            try:
                return float(raw)
            except ValueError:
                pass
    inst = (os.environ.get("RENDER_INSTANCE_TYPE") or "").lower()
    if "performance" in inst or inst.endswith("-8gb"):
        return 8192.0
    if "pro" in inst or "4gb" in inst:
        return 4096.0
    if os.environ.get("RENDER"):
        return 2048.0
    return None


def sample_memory_mb() -> Dict[str, Optional[float]]:
    """Sample current RSS/VMS; track process peak since import."""
    global _peak_rss_mb
    rss, vms = _read_proc_rss_vms_mb()
    if rss is None and resource is not None:
        try:
            usage = resource.getrusage(resource.RUSAGE_SELF)
            if platform.system() == "Darwin":
                rss = usage.ru_maxrss / (1024.0 * 1024.0)
            else:
                rss = usage.ru_maxrss / 1024.0
        except Exception:
            rss = None
    if rss is not None and rss > _peak_rss_mb:
        _peak_rss_mb = rss
    limit = memory_limit_mb()
    pct = round(100.0 * rss / limit, 1) if rss is not None and limit else None
    return {
        "rss_mb": round(rss, 1) if rss is not None else None,
        "vms_mb": round(vms, 1) if vms is not None else None,
        "peak_rss_mb": round(_peak_rss_mb, 1) if _peak_rss_mb else (round(rss, 1) if rss else None),
        "limit_mb": limit,
        "pct_of_limit": pct,
    }


def render_instance_context() -> Dict[str, Any]:
    return {
        "instance_id": (
            os.environ.get("RENDER_INSTANCE_ID")
            or os.environ.get("WORKER_ID")
            or socket.gethostname()
        ),
        "service_id": os.environ.get("RENDER_SERVICE_ID") or None,
        "service_name": os.environ.get("RENDER_SERVICE_NAME") or None,
        "region": os.environ.get("RENDER_REGION") or os.environ.get("AWS_REGION") or None,
        "git_commit": (os.environ.get("RENDER_GIT_COMMIT") or "")[:12] or None,
        "hostname": socket.gethostname(),
        "is_render": bool(os.environ.get("RENDER")),
    }


def worker_config_snapshot() -> Dict[str, Any]:
    return {
        "worker_lane": (os.environ.get("WORKER_LANE") or "full").strip().lower(),
        "worker_concurrency": int(os.environ.get("WORKER_CONCURRENCY", "3") or 3),
        "publish_concurrency": int(os.environ.get("PUBLISH_CONCURRENCY", "5") or 5),
        "heavy_pipeline_slots": int(os.environ.get("WORKER_HEAVY_PIPELINE_SLOTS", "1") or 1),
        "async_publish_queue": os.environ.get("ASYNC_PUBLISH_QUEUE", "false"),
        "worker_pipeline_profile": os.environ.get("WORKER_PIPELINE_PROFILE") or None,
    }


def format_semaphore_slots(total: int, semaphore) -> Dict[str, int]:
    """Return total/in_use/free for an asyncio.Semaphore."""
    if not total:
        return {"total": 0, "in_use": 0, "free": 0}
    try:
        free = int(getattr(semaphore, "_value", 0) or 0)
    except (TypeError, ValueError):
        free = 0
    free = max(0, min(total, free))
    in_use = max(0, total - free)
    return {"total": total, "in_use": in_use, "free": free}
