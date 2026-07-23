"""Render.com API client for maximum live worker/service observability.

Canonical env (set these on the **API** web service only):

* ``RENDER_MONITOR_API_KEY`` — Render account API key (``rnd_…``)
* ``RENDER_MONITOR_SERVICE_ID`` — worker service id to watch (``srv-…``)
* ``RENDER_MONITOR_OWNER_ID`` — workspace/owner id for OOM logs (``tea_…``, optional)

Legacy aliases still work: ``RENDER_API_KEY``, ``RENDER_WORKER_SERVICE_ID``,
``RENDER_OWNER_ID``. Do **not** manually set platform autos
``RENDER_SERVICE_ID`` / ``RENDER_INSTANCE_ID`` — Render injects those.

Read-only: service, events, deploys, instances, CPU/memory metrics, OOM log lines.
"""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger("uploadm8.render_platform")

RENDER_API_BASE = (os.environ.get("RENDER_API_BASE") or "https://api.render.com/v1").rstrip("/")

# Canonical names (preferred) — keep messages/docs on these only.
ENV_MONITOR_API_KEY = "RENDER_MONITOR_API_KEY"
ENV_MONITOR_SERVICE_ID = "RENDER_MONITOR_SERVICE_ID"
ENV_MONITOR_OWNER_ID = "RENDER_MONITOR_OWNER_ID"

CRITICAL_EVENT_TYPES = frozenset(
    {
        "server_failed",
        "server_hardware_failure",
        "server_restarted",
        "image_pull_failed",
        "service_suspended",
        "pipeline_minutes_exhausted",
    }
)

AUTOSCALE_EVENT_TYPES = frozenset(
    {
        "autoscaling_started",
        "autoscaling_ended",
        "autoscaling_config_changed",
        "instance_count_changed",
    }
)

OOM_LOG_TEXT_FILTERS = ("OOM", "Killed", "out of memory", "MemoryError", "Cannot allocate")


def _first_env(*keys: str) -> str:
    for key in keys:
        val = (os.environ.get(key) or "").strip()
        if val:
            return val
    return ""


def render_api_key() -> str:
    """Account API key for live monitoring (canonical: RENDER_MONITOR_API_KEY)."""
    return _first_env(ENV_MONITOR_API_KEY, "RENDER_API_KEY")


def render_worker_service_id() -> str:
    """Worker ``srv-…`` to monitor (canonical: RENDER_MONITOR_SERVICE_ID).

    Intentionally does **not** fall back to platform ``RENDER_SERVICE_ID`` —
    on the API that is the web service id, not the worker.
    """
    return _first_env(ENV_MONITOR_SERVICE_ID, "RENDER_WORKER_SERVICE_ID")


def render_owner_id() -> str:
    """Workspace owner id for log queries (canonical: RENDER_MONITOR_OWNER_ID)."""
    return _first_env(ENV_MONITOR_OWNER_ID, "RENDER_OWNER_ID")


def render_api_configured() -> bool:
    return bool(render_api_key() and render_worker_service_id())


def render_monitor_env_help() -> str:
    return (
        f"Set {ENV_MONITOR_API_KEY}=rnd_… and {ENV_MONITOR_SERVICE_ID}=srv-… "
        f"on the API service (+ optional {ENV_MONITOR_OWNER_ID}=tea_… for OOM logs)."
    )


def _auth_headers() -> Dict[str, str]:
    return {
        "Accept": "application/json",
        "Authorization": f"Bearer {render_api_key()}",
    }


def _unwrap_list_payload(payload: Any) -> List[dict]:
    """Render list endpoints return ``[{cursor, service|event|deploy|instance|…}, …]``."""
    if isinstance(payload, dict):
        for key in ("logs", "items", "data", "results"):
            nested = payload.get(key)
            if isinstance(nested, list):
                payload = nested
                break
        else:
            return []
    if not isinstance(payload, list):
        return []
    out: List[dict] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        for key in ("event", "service", "deploy", "job", "instance", "log"):
            nested = item.get(key)
            if isinstance(nested, dict):
                row = dict(nested)
                if item.get("cursor"):
                    row["_cursor"] = item["cursor"]
                out.append(row)
                break
        else:
            out.append(item)
    return out


def classify_event_severity(event_type: str) -> str:
    et = (event_type or "").strip().lower()
    if et in CRITICAL_EVENT_TYPES:
        return "critical"
    if et in AUTOSCALE_EVENT_TYPES:
        return "info"
    if et in ("deploy_ended", "deploy_started", "build_ended", "build_started"):
        return "info"
    return "warning" if et else "unknown"


async def _get_json(
    client: httpx.AsyncClient,
    path: str,
    *,
    params: Optional[List[Tuple[str, str]]] = None,
) -> Dict[str, Any]:
    url = f"{RENDER_API_BASE}{path}"
    resp = await client.get(url, headers=_auth_headers(), params=params or [])
    if resp.status_code >= 400:
        return {
            "ok": False,
            "error": f"Render API HTTP {resp.status_code}",
            "body": (resp.text or "")[:500],
            "status_code": resp.status_code,
        }
    try:
        data = resp.json()
    except Exception as e:
        return {"ok": False, "error": f"invalid JSON: {e}"}
    return {"ok": True, "data": data}


async def fetch_service(service_id: Optional[str] = None) -> Dict[str, Any]:
    sid = (service_id or render_worker_service_id()).strip()
    if not render_api_key():
        return {"ok": False, "configured": False, "error": f"{ENV_MONITOR_API_KEY} not set"}
    if not sid:
        return {
            "ok": False,
            "configured": False,
            "error": f"{ENV_MONITOR_SERVICE_ID} not set",
        }
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            res = await _get_json(client, f"/services/{sid}")
        if not res.get("ok"):
            return {"ok": False, "configured": True, "service_id": sid, "error": res.get("error")}
        data = res["data"]
        service = data.get("service") if isinstance(data, dict) and "service" in data else data
        return {"ok": True, "configured": True, "service_id": sid, "service": service}
    except Exception as e:
        logger.warning("render fetch_service failed: %s", e)
        return {"ok": False, "configured": True, "error": str(e)[:300]}


async def fetch_service_events(
    service_id: Optional[str] = None,
    *,
    limit: int = 20,
    event_types: Optional[List[str]] = None,
    start_time: Optional[str] = None,
    client: Optional[httpx.AsyncClient] = None,
) -> Dict[str, Any]:
    sid = (service_id or render_worker_service_id()).strip()
    if not render_api_key() or not sid:
        return {
            "ok": False,
            "configured": bool(render_api_key() and sid),
            "events": [],
            "error": "Render API not configured",
        }
    types = event_types or sorted(CRITICAL_EVENT_TYPES | AUTOSCALE_EVENT_TYPES | {"deploy_ended", "deploy_started"})
    q: List[Tuple[str, str]] = [("limit", str(max(1, min(100, int(limit or 20)))))]
    if start_time:
        q.append(("startTime", start_time))
    for t in types:
        q.append(("type", t))

    async def _run(c: httpx.AsyncClient) -> Dict[str, Any]:
        res = await _get_json(c, f"/services/{sid}/events", params=q)
        if not res.get("ok"):
            return {"ok": False, "configured": True, "events": [], "error": res.get("error"), "service_id": sid}
        raw = _unwrap_list_payload(res["data"])
        events: List[dict] = []
        for row in raw:
            et = str(row.get("type") or row.get("eventType") or "")
            events.append(
                {
                    "id": row.get("id"),
                    "type": et,
                    "severity": classify_event_severity(et),
                    "timestamp": row.get("timestamp") or row.get("createdAt") or row.get("time"),
                    "details": {
                        k: v
                        for k, v in row.items()
                        if k
                        not in (
                            "id",
                            "type",
                            "eventType",
                            "timestamp",
                            "createdAt",
                            "time",
                            "_cursor",
                        )
                    },
                }
            )
        return {"ok": True, "configured": True, "service_id": sid, "events": events}

    try:
        if client is not None:
            return await _run(client)
        async with httpx.AsyncClient(timeout=20.0) as c:
            return await _run(c)
    except Exception as e:
        logger.warning("render fetch_service_events failed: %s", e)
        return {"ok": False, "configured": True, "events": [], "error": str(e)[:300]}


async def fetch_deploys(
    service_id: Optional[str] = None,
    *,
    limit: int = 5,
    client: Optional[httpx.AsyncClient] = None,
) -> Dict[str, Any]:
    sid = (service_id or render_worker_service_id()).strip()
    if not render_api_key() or not sid:
        return {"ok": False, "deploys": [], "error": "not configured"}

    async def _run(c: httpx.AsyncClient) -> Dict[str, Any]:
        res = await _get_json(
            c,
            f"/services/{sid}/deploys",
            params=[("limit", str(max(1, min(20, int(limit or 5)))))],
        )
        if not res.get("ok"):
            return {"ok": False, "deploys": [], "error": res.get("error")}
        raw = _unwrap_list_payload(res["data"])
        deploys = []
        for row in raw:
            commit = row.get("commit") if isinstance(row.get("commit"), dict) else {}
            deploys.append(
                {
                    "id": row.get("id"),
                    "status": row.get("status"),
                    "trigger": row.get("trigger"),
                    "created_at": row.get("createdAt"),
                    "finished_at": row.get("finishedAt"),
                    "commit_id": (commit.get("id") or "")[:12] or None,
                    "commit_message": (commit.get("message") or "")[:120] or None,
                }
            )
        return {"ok": True, "deploys": deploys, "latest": deploys[0] if deploys else None}

    try:
        if client is not None:
            return await _run(client)
        async with httpx.AsyncClient(timeout=20.0) as c:
            return await _run(c)
    except Exception as e:
        return {"ok": False, "deploys": [], "error": str(e)[:300]}


async def fetch_instances(
    service_id: Optional[str] = None,
    *,
    client: Optional[httpx.AsyncClient] = None,
) -> Dict[str, Any]:
    sid = (service_id or render_worker_service_id()).strip()
    if not render_api_key() or not sid:
        return {"ok": False, "instances": [], "error": "not configured"}

    async def _run(c: httpx.AsyncClient) -> Dict[str, Any]:
        res = await _get_json(c, f"/services/{sid}/instances")
        if not res.get("ok"):
            return {"ok": False, "instances": [], "error": res.get("error")}
        raw = _unwrap_list_payload(res["data"])
        instances = []
        for row in raw:
            st = str(row.get("status") or row.get("state") or "").lower()
            instances.append(
                {
                    "id": row.get("id"),
                    "status": st or row.get("status"),
                    "created_at": row.get("createdAt"),
                }
            )
        running = sum(
            1
            for i in instances
            if str(i.get("status") or "").lower() in ("running", "active", "up", "ok")
        )
        return {
            "ok": True,
            "instances": instances,
            "instance_count": len(instances),
            "running_count": running or len(instances),
        }

    try:
        if client is not None:
            return await _run(client)
        async with httpx.AsyncClient(timeout=20.0) as c:
            return await _run(c)
    except Exception as e:
        return {"ok": False, "instances": [], "error": str(e)[:300]}


def summarize_metric_timeseries(payload: Any) -> Dict[str, Any]:
    """Reduce Render metrics JSON to latest/max over the window."""
    series_list: List[dict] = []
    if isinstance(payload, dict):
        series_list = payload.get("timeseries") or payload.get("series") or []
        if not series_list and "values" in payload:
            series_list = [payload]
    elif isinstance(payload, list):
        series_list = payload

    latest = None
    max_v = None
    points = 0
    for series in series_list:
        if not isinstance(series, dict):
            continue
        values = series.get("values") or series.get("data") or []
        for pt in values:
            if isinstance(pt, dict):
                v = pt.get("value")
                if v is None:
                    v = pt.get("Value")
            elif isinstance(pt, (list, tuple)) and len(pt) >= 2:
                v = pt[1]
            else:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            points += 1
            latest = fv
            max_v = fv if max_v is None else max(max_v, fv)
    return {
        "latest": round(latest, 4) if latest is not None else None,
        "max": round(max_v, 4) if max_v is not None else None,
        "points": points,
    }


async def fetch_resource_metrics(
    service_id: Optional[str] = None,
    *,
    window_minutes: int = 30,
    client: Optional[httpx.AsyncClient] = None,
) -> Dict[str, Any]:
    """GET /metrics/cpu and /metrics/memory for the worker service."""
    sid = (service_id or render_worker_service_id()).strip()
    if not render_api_key() or not sid:
        return {"ok": False, "error": "not configured"}

    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=max(5, int(window_minutes or 30)))
    base_q = [
        ("resource", sid),
        ("startTime", start.isoformat().replace("+00:00", "Z")),
        ("endTime", end.isoformat().replace("+00:00", "Z")),
        ("resolutionSeconds", "60"),
        ("aggregationMethod", "MAX"),
    ]

    async def _run(c: httpx.AsyncClient) -> Dict[str, Any]:
        cpu_res, mem_res = await asyncio.gather(
            _get_json(c, "/metrics/cpu", params=base_q),
            _get_json(c, "/metrics/memory", params=base_q),
        )
        out: Dict[str, Any] = {
            "ok": bool(cpu_res.get("ok") or mem_res.get("ok")),
            "window_minutes": window_minutes,
            "cpu": summarize_metric_timeseries(cpu_res.get("data")) if cpu_res.get("ok") else None,
            "memory": summarize_metric_timeseries(mem_res.get("data")) if mem_res.get("ok") else None,
            "cpu_error": None if cpu_res.get("ok") else cpu_res.get("error"),
            "memory_error": None if mem_res.get("ok") else mem_res.get("error"),
        }
        return out

    try:
        if client is not None:
            return await _run(client)
        async with httpx.AsyncClient(timeout=25.0) as c:
            return await _run(c)
    except Exception as e:
        return {"ok": False, "error": str(e)[:300]}


async def fetch_oom_logs(
    service_id: Optional[str] = None,
    *,
    owner_id: Optional[str] = None,
    limit: int = 15,
    client: Optional[httpx.AsyncClient] = None,
) -> Dict[str, Any]:
    """GET /logs filtered for OOM / Killed lines (needs workspace ownerId)."""
    sid = (service_id or render_worker_service_id()).strip()
    oid = (owner_id or render_owner_id()).strip()
    if not render_api_key() or not sid:
        return {"ok": False, "logs": [], "error": "not configured"}
    if not oid:
        return {
            "ok": False,
            "logs": [],
            "error": f"{ENV_MONITOR_OWNER_ID} not set (and not found on service)",
            "skipped": True,
        }

    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=6)
    q: List[Tuple[str, str]] = [
        ("ownerId", oid),
        ("resource", sid),
        ("direction", "backward"),
        ("limit", str(max(1, min(50, int(limit or 15))))),
        ("startTime", start.isoformat().replace("+00:00", "Z")),
        ("endTime", end.isoformat().replace("+00:00", "Z")),
        ("type", "app"),
    ]
    for t in OOM_LOG_TEXT_FILTERS:
        q.append(("text", t))

    async def _run(c: httpx.AsyncClient) -> Dict[str, Any]:
        res = await _get_json(c, "/logs", params=q)
        if not res.get("ok"):
            return {"ok": False, "logs": [], "error": res.get("error")}
        raw = _unwrap_list_payload(res["data"])
        logs = []
        for row in raw[: limit or 15]:
            logs.append(
                {
                    "timestamp": row.get("timestamp") or row.get("time"),
                    "message": (row.get("message") or row.get("text") or row.get("body") or "")[:400],
                    "level": row.get("level"),
                    "instance": row.get("instance") or row.get("instanceId"),
                }
            )
        return {"ok": True, "logs": logs, "count": len(logs), "owner_id": oid}

    try:
        if client is not None:
            return await _run(client)
        async with httpx.AsyncClient(timeout=25.0) as c:
            return await _run(c)
    except Exception as e:
        return {"ok": False, "logs": [], "error": str(e)[:300]}


def summarize_render_events(events: List[dict]) -> Dict[str, Any]:
    critical = sum(1 for e in events if e.get("severity") == "critical")
    autoscale = sum(1 for e in events if (e.get("type") or "") in AUTOSCALE_EVENT_TYPES)
    latest_critical = next((e for e in events if e.get("severity") == "critical"), None)
    return {
        "event_count": len(events),
        "critical_count": critical,
        "autoscale_count": autoscale,
        "latest_critical": latest_critical,
        "has_recent_failure": critical > 0,
    }


def build_observability_health(
    *,
    events_summary: Dict[str, Any],
    instances: Optional[Dict[str, Any]],
    metrics: Optional[Dict[str, Any]],
    deploys: Optional[Dict[str, Any]],
    oom_logs: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Single traffic-light for admin UI."""
    score = 100
    flags: List[str] = []
    if events_summary.get("has_recent_failure"):
        score -= 40
        flags.append("recent_render_failure")
    running = int((instances or {}).get("running_count") or 0)
    if instances and instances.get("ok") and running == 0:
        score -= 35
        flags.append("no_running_instances")
    cpu_max = ((metrics or {}).get("cpu") or {}).get("max")
    if cpu_max is not None and cpu_max >= 0.9:
        score -= 15
        flags.append("cpu_hot")
    mem_latest = ((metrics or {}).get("memory") or {}).get("latest")
    # Render memory metrics are often bytes; treat >1.7GB on 2GB as hot if absolute.
    if mem_latest is not None and mem_latest > 1.7 * 1024**3:
        score -= 15
        flags.append("platform_memory_hot")
    if (oom_logs or {}).get("count"):
        score -= 25
        flags.append("oom_log_hits")
    latest_deploy = (deploys or {}).get("latest") or {}
    if str(latest_deploy.get("status") or "").lower() in ("build_failed", "update_failed", "failed", "canceled"):
        score -= 10
        flags.append("deploy_failed")
    score = max(0, score)
    if score >= 80:
        status = "healthy"
    elif score >= 50:
        status = "degraded"
    else:
        status = "critical"
    return {"status": status, "score": score, "flags": flags}


_LIVE_CACHE: Dict[str, Any] = {"key": None, "expires": 0.0, "payload": None}


def _live_cache_ttl_sec() -> float:
    try:
        return max(5.0, float(os.environ.get("RENDER_MONITOR_CACHE_TTL_SEC", "30") or 30))
    except ValueError:
        return 30.0


async def build_render_live_snapshot(
    *,
    service_id: Optional[str] = None,
    event_limit: int = 20,
    include_logs: bool = True,
    include_metrics: bool = True,
    use_cache: bool = True,
    event_lookback_minutes: Optional[int] = None,
) -> Dict[str, Any]:
    """Full live observability bundle for admin/ops (TTL-cached to avoid Render rate limits)."""
    import time as _time

    if not render_api_configured():
        return {
            "ok": False,
            "configured": False,
            "message": (
                render_monitor_env_help()
                + " Then Admin KPI streams live events, deploys, instances, metrics, and OOM logs."
            ),
            "service": None,
            "events": [],
            "deploys": [],
            "instances": None,
            "metrics": None,
            "oom_logs": None,
            "summary": summarize_render_events([]),
            "health": {"status": "unknown", "score": 0, "flags": ["not_configured"]},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    sid = (service_id or render_worker_service_id()).strip()
    try:
        lookback = int(
            event_lookback_minutes
            if event_lookback_minutes is not None
            else (os.environ.get("WATCHDOG_RENDER_LOOKBACK_MIN") or 30)
        )
    except ValueError:
        lookback = 30
    lookback = max(5, min(180, lookback))
    event_start = (
        datetime.now(timezone.utc) - timedelta(minutes=lookback)
    ).isoformat().replace("+00:00", "Z")
    cache_key = f"{sid}:{event_limit}:{int(include_logs)}:{int(include_metrics)}:{lookback}"
    now = _time.time()
    if (
        use_cache
        and _LIVE_CACHE.get("key") == cache_key
        and float(_LIVE_CACHE.get("expires") or 0) > now
        and isinstance(_LIVE_CACHE.get("payload"), dict)
    ):
        cached = dict(_LIVE_CACHE["payload"])
        cached["cached"] = True
        return cached

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            service_task = _get_json(client, f"/services/{sid}")
            events_task = fetch_service_events(
                sid, limit=event_limit, start_time=event_start, client=client
            )
            deploys_task = fetch_deploys(sid, limit=5, client=client)
            instances_task = fetch_instances(sid, client=client)
            metrics_task = (
                fetch_resource_metrics(sid, window_minutes=30, client=client)
                if include_metrics
                else asyncio.sleep(0, result={"ok": False, "skipped": True})
            )

            service_raw, events_res, deploys_res, instances_res, metrics_res = await asyncio.gather(
                service_task,
                events_task,
                deploys_task,
                instances_task,
                metrics_task,
            )

            service = None
            service_error = None
            owner = render_owner_id()
            if service_raw.get("ok"):
                data = service_raw["data"]
                service = data.get("service") if isinstance(data, dict) and "service" in data else data
                if isinstance(service, dict):
                    owner = owner or str(service.get("ownerId") or service.get("owner_id") or "")
            else:
                service_error = service_raw.get("error")

            oom_res: Dict[str, Any]
            if include_logs:
                oom_res = await fetch_oom_logs(sid, owner_id=owner or None, limit=12, client=client)
            else:
                oom_res = {"ok": False, "logs": [], "skipped": True}

        events = events_res.get("events") or []
        summary = summarize_render_events(events)
        health = build_observability_health(
            events_summary=summary,
            instances=instances_res if instances_res.get("ok") else None,
            metrics=metrics_res if metrics_res.get("ok") else None,
            deploys=deploys_res if deploys_res.get("ok") else None,
            oom_logs=oom_res if oom_res.get("ok") else None,
        )
        payload = {
            "ok": True,
            "configured": True,
            "service_id": sid,
            "service": service,
            "service_error": service_error,
            "events": events,
            "events_error": None if events_res.get("ok") else events_res.get("error"),
            "deploys": deploys_res.get("deploys") or [],
            "latest_deploy": deploys_res.get("latest"),
            "deploys_error": None if deploys_res.get("ok") else deploys_res.get("error"),
            "instances": instances_res if instances_res.get("ok") else None,
            "instances_error": None if instances_res.get("ok") else instances_res.get("error"),
            "metrics": metrics_res if metrics_res.get("ok") else None,
            "metrics_error": None if metrics_res.get("ok") else metrics_res.get("error"),
            "oom_logs": oom_res,
            "summary": summary,
            "health": health,
            "cached": False,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if use_cache:
            _LIVE_CACHE["key"] = cache_key
            _LIVE_CACHE["expires"] = now + _live_cache_ttl_sec()
            _LIVE_CACHE["payload"] = payload
        return payload
    except Exception as e:
        logger.warning("build_render_live_snapshot failed: %s", e)
        return {
            "ok": False,
            "configured": True,
            "error": str(e)[:300],
            "events": [],
            "summary": summarize_render_events([]),
            "health": {"status": "unknown", "score": 0, "flags": ["fetch_error"]},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
