"""Process memory and Render instance metadata for worker/API observability."""
from __future__ import annotations

import os
import platform
import socket
from typing import Any, Dict, Literal, Optional

try:
    import resource
except ImportError:
    resource = None  # Windows — use /proc only (unavailable locally)

_peak_rss_mb: float = 0.0

MemoryPressure = Literal["ok", "soft", "hard"]


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


def _cgroup_memory_limit_mb() -> Optional[float]:
    """Best-effort container memory limit (cgroup v1/v2) — accurate on Render."""
    paths = (
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    )
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = (f.read() or "").strip()
            if not raw or raw.lower() == "max":
                continue
            n = int(raw)
            # Ignore absurd "unlimited" sentinel values from cgroup v1.
            if n <= 0 or n >= (1 << 60):
                continue
            return n / (1024.0 * 1024.0)
        except (OSError, ValueError):
            continue
    return None


def memory_limit_mb() -> Optional[float]:
    for key in ("RENDER_MEMORY_LIMIT_MB", "MEMORY_LIMIT_MB", "WEB_MEMORY_LIMIT_MB"):
        raw = (os.environ.get(key) or "").strip()
        if raw:
            try:
                return float(raw)
            except ValueError:
                pass
    cg = _cgroup_memory_limit_mb()
    if cg is not None:
        return cg
    inst = (os.environ.get("RENDER_INSTANCE_TYPE") or "").lower()
    if "performance" in inst or inst.endswith("-8gb"):
        return 8192.0
    if "pro" in inst or "4gb" in inst:
        return 4096.0
    if "starter" in inst or "512" in inst:
        return 512.0
    if os.environ.get("RENDER"):
        return 2048.0
    return None


def memory_admit_pct() -> float:
    try:
        return float(os.environ.get("MEMORY_ADMIT_PCT", "75") or 75)
    except ValueError:
        return 75.0


def memory_hard_pct() -> float:
    try:
        return float(os.environ.get("MEMORY_HARD_PCT", "88") or 88)
    except ValueError:
        return 88.0


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


def memory_pressure_level(sample: Optional[Dict[str, Optional[float]]] = None) -> MemoryPressure:
    """ok / soft (pause new heavy work) / hard (requeue, never start another encode)."""
    mem = sample if sample is not None else sample_memory_mb()
    pct = mem.get("pct_of_limit")
    if pct is None:
        return "ok"
    if pct >= memory_hard_pct():
        return "hard"
    if pct >= memory_admit_pct():
        return "soft"
    return "ok"


def blocks_new_process_job(sample: Optional[Dict[str, Optional[float]]] = None) -> bool:
    """True when starting another FFmpeg-heavy job would risk Render OOM kill."""
    return memory_pressure_level(sample) != "ok"


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
