"""
Optional Sentry SDK initialization when SENTRY_DSN is set.

DSN and auth tokens are secrets — configure only via the host environment or secret store,
never commit them to the repo.
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any


# Transient errors that occur during lifespan shutdown / pre-startup races.
# These are not actionable — the request happened to land while the connection
# pool was being torn down (Ctrl+C, hot-reload, deploy) or before lifespan
# finished initializing it. Filtering keeps Sentry signal high.
def _should_drop_async_shutdown_noise(event: dict[str, Any], hint: dict[str, Any] | None) -> bool:
    """
    Ctrl+C / deploy / handler teardown surface as CancelledError or KeyboardInterrupt.
    Neither is actionable in Sentry for this API.
    """
    info = hint.get("exc_info") if hint else None
    if not info or len(info) < 1 or info[0] is None:
        return False
    etype = info[0]
    if etype is KeyboardInterrupt:
        return True
    try:
        if issubclass(etype, asyncio.CancelledError):
            # Almost always shutdown / task teardown (uvicorn lifespan, handler cancellation).
            # Production stacks may not include "uvicorn" in the exception chain — still not actionable.
            return True
    except TypeError:
        pass
    return False


# asyncio logs this when a Task is GC'd while still pending (reload, process exit,
# or orphaned fire-and-forget coroutines). Not actionable as a Sentry "error".
_ASYNCIO_PENDING_TASK_DESTROYED = "Task was destroyed but it is pending"


def _is_dev_like_sentry_env() -> bool:
    v = (_env_name() or "").strip().lower()
    return v in ("development", "dev", "local", "test")


def _should_drop_dev_worker_ffmpeg_watermark(event: dict[str, Any], hint: dict[str, Any] | None) -> bool:
    """Dev laptops often run the worker without ffmpeg; skip Sentry noise for that case."""
    if not _is_dev_like_sentry_env():
        return False
    try:
        for v in _exception_chain_values(event, hint):
            vl = v.lower()
            if "ffmpeg" not in vl:
                continue
            if "watermark" in vl or "could not be applied" in vl:
                return True
    except Exception:
        pass
    return False


def _should_drop_fastapi_testclient_request(event: dict[str, Any]) -> bool:
    """
    Starlette/FastAPI ``TestClient`` uses host ``testserver`` (see ASGI scope).

    Pytest API tests that intentionally raise ``HTTPException`` (e.g. missing
    ``PIKZELS_API_KEY``) should not create Sentry issues in the production project
    when ``SENTRY_DSN`` is set in CI or local dev.
    """
    try:
        req = event.get("request") or {}
        url = str(req.get("url") or "")
        if "testserver" in url.lower():
            return True
    except Exception:
        pass
    return False


def _should_drop_uvicorn_lifespan_shutdown_event(event: dict[str, Any]) -> bool:
    """
    Uvicorn logs ``CancelledError`` / ``KeyboardInterrupt`` from the lifespan ASGI
    channel when the process stops (Ctrl+C, deploy, hot reload). Those are not
    actionable bugs; Sentry often receives them via the logging integration without
    ``hint['exc_info']``, so they bypass ``_should_drop_async_shutdown_noise``.
    """
    try:
        log_name = str(event.get("logger") or "").lower()
        if "uvicorn" not in log_name:
            return False
        for entry in (event.get("exception") or {}).get("values", []) or []:
            t = str(entry.get("type") or "")
            if "CancelledError" in t or "KeyboardInterrupt" in t:
                return True
        le = event.get("logentry") or {}
        if isinstance(le, dict):
            blob = f"{le.get('message') or ''} {le.get('formatted') or ''}".lower()
            if "cancellederror" in blob or "keyboardinterrupt" in blob:
                return True
    except Exception:
        pass
    return False


def _should_drop_asyncio_pending_task_log(event: dict[str, Any]) -> bool:
    try:
        le = event.get("logentry") or {}
        parts: list[str] = []
        if isinstance(le, dict):
            parts.extend([str(le.get("message") or ""), str(le.get("formatted") or "")])
        parts.append(str(event.get("message") or ""))
        return _ASYNCIO_PENDING_TASK_DESTROYED in " ".join(parts)
    except Exception:
        return False


_DROPPED_EXC_VALUES: tuple[str, ...] = (
    "'NoneType' object has no attribute 'acquire'",
    "'NoneType' object has no attribute 'fetchrow'",
    "'NoneType' object has no attribute 'fetch'",
    "'NoneType' object has no attribute 'fetchval'",
    "'NoneType' object has no attribute 'execute'",
    "unexpected connection_lost() call",
)


def _should_drop_localhost_db_acquire_noise(event: dict[str, Any], hint: dict[str, Any] | None) -> bool:
    """
    Pool saturation / Neon socket drops during local overnight E2E become clean
    HTTP 503s. Reporting them as unhandled TimeoutError (UPLOADM8-7H/7J/7N) or
    asyncio connection_lost noise (UPLOADM8-7P) drowns real signal.
    """
    info = hint.get("exc_info") if hint else None
    etype = None
    if info and len(info) >= 1 and info[0] is not None:
        etype = info[0]
    types: list[str] = []
    if etype is not None:
        try:
            types.append(etype.__name__)
        except Exception:
            pass
    for entry in (event.get("exception") or {}).get("values", []) or []:
        t = str(entry.get("type") or "")
        if t:
            types.append(t)
    type_blob = " ".join(types)
    # Expected fail-fast after acquire retries — never actionable as an error issue.
    if "DatabaseUnavailableError" in type_blob:
        return True
    le = event.get("logentry") or {}
    log_blob = ""
    if isinstance(le, dict):
        log_blob = f"{le.get('message') or ''} {le.get('formatted') or ''}"
    if "unexpected connection_lost()" in log_blob or "Future exception was never retrieved" in log_blob:
        return True
    if not _is_localhost_url(_request_url(event)) and not _is_local_uvicorn_process():
        return False
    if "TimeoutError" in type_blob:
        return True
    return False


def _exception_chain_values(event: dict[str, Any], hint: dict[str, Any] | None) -> list[str]:
    values: list[str] = []
    info = hint.get("exc_info") if hint else None
    if info and len(info) >= 2 and info[1] is not None:
        exc = info[1]
        seen = set()
        while exc is not None and id(exc) not in seen:
            seen.add(id(exc))
            try:
                values.append(str(exc))
            except Exception:
                pass
            exc = getattr(exc, "__cause__", None) or getattr(exc, "__context__", None)
    for entry in (event.get("exception") or {}).get("values", []) or []:
        v = entry.get("value")
        if isinstance(v, str):
            values.append(v)
    return values


def _before_send(event: dict[str, Any], hint: dict[str, Any] | None) -> dict[str, Any] | None:
    try:
        if _should_drop_async_shutdown_noise(event, hint):
            return None
        if _should_drop_uvicorn_lifespan_shutdown_event(event):
            return None
        if _should_drop_asyncio_pending_task_log(event):
            return None
        if _should_drop_dev_worker_ffmpeg_watermark(event, hint):
            return None
        if _should_drop_fastapi_testclient_request(event):
            return None
        if _should_drop_localhost_db_acquire_noise(event, hint):
            return None
        for v in _exception_chain_values(event, hint):
            for needle in _DROPPED_EXC_VALUES:
                if needle in v:
                    return None
    except Exception:
        pass
    return event


def _is_asyncpg_pool_reset_description(description: str) -> bool:
    """asyncpg pool release runs RESET ALL — not actionable app SQL (Sentry UPLOADM8-*)."""
    d = (description or "").lower()
    return (
        "pg_advisory_unlock_all" in d
        or "reset all" in d
        or "close all" in d
        or "unlisten *" in d
    )


def _filter_pool_reset_spans(spans: list) -> list:
    """Drop asyncpg pool-reset spans; recurse into nested children when present."""
    out: list = []
    for s in spans or []:
        if not isinstance(s, dict):
            continue
        desc = str(s.get("description") or "")
        if _is_asyncpg_pool_reset_description(desc):
            continue
        child = s.get("spans")
        if isinstance(child, list):
            filtered_child = _filter_pool_reset_spans(child)
            if filtered_child != child:
                s = dict(s)
                s["spans"] = filtered_child
        out.append(s)
    return out


def _strip_asyncpg_pool_reset_spans(event: dict[str, Any]) -> dict[str, Any]:
    """
    Remove asyncpg connection-pool reset spans from performance transactions.

    Sentry's consecutive-DB detector treats ``pg_advisory_unlock_all`` as a parallelizable
    SELECT (no WHERE clause) and opens issues like UPLOADM8-28 even when the handler uses
    a single ``pool.acquire()``. Stripping these spans keeps real query regressions visible.
    """
    spans = event.get("spans")
    if not isinstance(spans, list) or not spans:
        return event
    filtered = _filter_pool_reset_spans(spans)
    if filtered != spans:
        event = dict(event)
        event["spans"] = filtered
    return event


def _request_url(event: dict[str, Any]) -> str:
    try:
        req = event.get("request") or {}
        return str(req.get("url") or "")
    except Exception:
        return ""


def _is_localhost_url(url: str) -> bool:
    u = (url or "").lower()
    return (
        "127.0.0.1" in u
        or "://localhost" in u
        or "://[::1]" in u
        or "testserver" in u
    )


def _is_local_uvicorn_process() -> bool:
    """True when this process is clearly a loopback API (local E2E / laptop uvicorn)."""
    argv = " ".join(sys.argv).lower()
    if "127.0.0.1" in argv or "--host localhost" in argv or "--host ::1" in argv:
        return True
    e2e = (os.environ.get("E2E_BASE_URL") or "").strip().lower()
    if e2e and _is_localhost_url(e2e) and os.environ.get("E2E_TUP", "").lower() in (
        "1",
        "true",
        "yes",
    ):
        return True
    return False


def _should_drop_localhost_transaction(event: dict[str, Any]) -> bool:
    """
    Local E2E against 127.0.0.1 must not create/regress production Slow DB Query issues
    (UPLOADM8-6E / 5R / 30 / 5T). Neon wake latency looks like slow SELECTs under load.
    """
    raw = (os.environ.get("SENTRY_DROP_LOCALHOST_TRACES") or "1").strip().lower()
    if raw in ("0", "false", "no"):
        return False
    return _is_localhost_url(_request_url(event))


def _before_send_transaction(event: dict[str, Any], hint: dict[str, Any] | None) -> dict[str, Any] | None:
    try:
        if _should_drop_fastapi_testclient_request(event):
            return None
        if _should_drop_localhost_transaction(event):
            return None
        return _strip_asyncpg_pool_reset_spans(event)
    except Exception:
        return event


def _trace_sample_rate() -> float:
    # Local uvicorn: never sample performance traces into the production project.
    if _is_local_uvicorn_process():
        return 0.0
    raw = (os.environ.get("SENTRY_TRACES_SAMPLE_RATE") or os.environ.get("SENTRY_TRACES_RATE") or "0").strip()
    if not raw:
        return 0.0
    try:
        return float(raw)
    except ValueError:
        return 0.0


def _env_name() -> str | None:
    # Loopback API must never tag as production — .env often has SENTRY_ENVIRONMENT=production
    # for deploy, while laptop uvicorn still uses that same file for E2E.
    if _is_local_uvicorn_process():
        return "development"
    v = (
        os.environ.get("SENTRY_ENVIRONMENT")
        or os.environ.get("SENTRY_ENV")
        or os.environ.get("ENVIRONMENT")
        or ""
    ).strip()
    if v:
        # If BASE_URL / E2E clearly local, prefer development over a stale production tag.
        base = (os.environ.get("BASE_URL") or "").strip().lower()
        e2e = (os.environ.get("E2E_BASE_URL") or "").strip().lower()
        if v.lower() == "production" and (
            _is_localhost_url(base) or _is_localhost_url(e2e)
        ):
            return "development"
        return v
    base = (os.environ.get("BASE_URL") or "").strip().lower()
    if any(h in base for h in ("127.0.0.1", "localhost")):
        return "development"
    return None


def _release() -> str | None:
    v = (os.environ.get("SENTRY_RELEASE") or "").strip()
    return v or None


def init_sentry_for_api() -> None:
    dsn = (os.environ.get("SENTRY_DSN") or "").strip()
    if not dsn:
        return
    import sentry_sdk
    from sentry_sdk.integrations.fastapi import FastApiIntegration
    from sentry_sdk.integrations.starlette import StarletteIntegration

    sentry_sdk.init(
        dsn=dsn,
        integrations=[
            StarletteIntegration(),
            FastApiIntegration(),
        ],
        traces_sample_rate=_trace_sample_rate(),
        environment=_env_name(),
        release=_release(),
        before_send=_before_send,
        before_send_transaction=_before_send_transaction,
    )


def init_sentry_for_worker() -> None:
    dsn = (os.environ.get("SENTRY_DSN") or "").strip()
    if not dsn:
        return
    import sentry_sdk
    from sentry_sdk.integrations.asyncio import AsyncioIntegration

    sentry_sdk.init(
        dsn=dsn,
        integrations=[AsyncioIntegration()],
        traces_sample_rate=_trace_sample_rate(),
        environment=_env_name(),
        release=_release(),
        before_send=_before_send,
    )
