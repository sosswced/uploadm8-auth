"""Build parametrized GET smoke cases from /openapi.json."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
import os
from typing import Any, Iterable

import httpx

SKIP_PATH_RE = re.compile(
    r"(webhook|oauth/callback|stripe|/auth/register|/auth/login|/auth/refresh|"
    r"/auth/logout|/auth/forgot|/auth/reset|/healthz|/metrics|session-probe)",
    re.I,
)

# Heavy reads (LLM, HF, large aggregates) — opt-in via E2E_INCLUDE_SLOW_API=1 overnight.
SLOW_PATH_RE = re.compile(
    r"(/coach|optimize-preview|/marketing/intel|/marketing/reports|"
    r"/marketing/ai/truth|upload-ai-trace|sync-analytics|/ml/loop-reports|"
    r"/admin/kpi|/admin/kpis|platform-metrics(?!/cached)|"
    r"smart-insights|ai-insights|pikzels|thumbnail-studio/jobs|content-insights)",
    re.I,
)

# GETs that are known-safe reads for master admin overnight sweeps.
SAFE_GET_PREFIXES: tuple[str, ...] = (
    "/api/me",
    "/api/admin",
    "/api/uploads",
    "/api/entitlements",
    "/api/wallet",
    "/api/settings",
    "/api/dashboard",
    "/api/analytics",
    "/api/catalog",
    "/api/billing",
    "/api/platforms",
    "/api/groups",
    "/api/scheduled/list",
    "/api/scheduling",
    "/api/preferences",
    "/api/thumbnail-studio",
    "/api/trill",
    "/api/features",
    "/api/shell",
    "/api/ops",
    "/api/support",
    "/api/workspace",
    "/api/marketing",
)

ACCEPTABLE_STATUSES = frozenset({200, 204, 400, 401, 403, 404, 405, 422})


@dataclass(frozen=True)
class ApiGetCase:
    method: str
    path: str
    summary: str

    @property
    def id(self) -> str:
        return f"{self.method} {self.path}"


def _substitute_path_params(path: str, ctx: dict[str, str]) -> str | None:
    out = path
    for match in re.finditer(r"\{([^}]+)\}", path):
        key = match.group(1)
        val = ctx.get(key) or ctx.get(key.replace("-", "_"))
        if not val:
            # Generic UUID placeholder — 404 is acceptable overnight.
            if key.endswith("_id") or key in ("id", "upload_id", "user_id", "job_id"):
                val = ctx.get("_default_uuid", str(uuid.UUID(int=0)))
            else:
                return None
        out = out.replace("{" + key + "}", val)
    return out


def build_context(client: httpx.Client) -> dict[str, str]:
    ctx: dict[str, str] = {"_default_uuid": "00000000-0000-0000-0000-000000000001"}
    try:
        me = client.get("/api/me")
        if me.status_code == 200:
            body = me.json()
            uid = str(body.get("id") or body.get("user_id") or "")
            if uid:
                ctx["user_id"] = uid
                ctx["id"] = uid
    except Exception:
        pass
    try:
        uploads = client.get("/api/uploads", params={"limit": 1})
        if uploads.status_code == 200:
            rows = uploads.json()
            if isinstance(rows, list) and rows:
                uid = str(rows[0].get("id") or rows[0].get("upload_id") or "")
                if uid:
                    ctx["upload_id"] = uid
                    ctx["id"] = uid
            elif isinstance(rows, dict):
                items = rows.get("uploads") or rows.get("items") or []
                if items:
                    uid = str(items[0].get("id") or "")
                    if uid:
                        ctx["upload_id"] = uid
    except Exception:
        pass
    return ctx


def fetch_openapi(client: httpx.Client) -> dict[str, Any]:
    r = client.get("/openapi.json")
    r.raise_for_status()
    return r.json()


def include_slow_api_paths() -> bool:
    return os.environ.get("E2E_INCLUDE_SLOW_API", "").lower() in ("1", "true", "yes")


def iter_read_smoke_cases(
    openapi: dict[str, Any],
    *,
    ctx: dict[str, str],
    safe_only: bool = True,
) -> Iterable[ApiGetCase]:
    paths = openapi.get("paths") or {}
    for path, methods in sorted(paths.items()):
        if SKIP_PATH_RE.search(path):
            continue
        if not include_slow_api_paths() and SLOW_PATH_RE.search(path):
            continue
        if safe_only and not any(path.startswith(p) for p in SAFE_GET_PREFIXES):
            continue
        if not isinstance(methods, dict):
            continue
        for method, meta in methods.items():
            m = method.upper()
            if m != "GET":
                continue
            resolved = _substitute_path_params(path, ctx)
            if resolved is None:
                continue
            summary = ""
            if isinstance(meta, dict):
                summary = str(meta.get("summary") or meta.get("operationId") or "")
            yield ApiGetCase(method=m, path=resolved, summary=summary)


def assert_acceptable_status(status: int, case: ApiGetCase) -> None:
    if status in ACCEPTABLE_STATUSES:
        return
    if status >= 500:
        raise AssertionError(f"{case.id} returned server error {status}")
    # Unexpected 3xx without follow — flag it.
    if 300 <= status < 400:
        raise AssertionError(f"{case.id} returned redirect {status}")
