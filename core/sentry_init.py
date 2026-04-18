"""
Optional Sentry SDK initialization when SENTRY_DSN is set.

DSN and auth tokens are secrets — configure only via the host environment or secret store,
never commit them to the repo.
"""

from __future__ import annotations

import os


def _trace_sample_rate() -> float:
    raw = (os.environ.get("SENTRY_TRACES_SAMPLE_RATE") or "0").strip()
    if not raw:
        return 0.0
    try:
        return float(raw)
    except ValueError:
        return 0.0


def _env_name() -> str | None:
    v = (os.environ.get("SENTRY_ENVIRONMENT") or os.environ.get("ENVIRONMENT") or "").strip()
    return v or None


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
    )
