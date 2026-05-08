"""
Async client for Pikzels public API v2 (https://api.pikzels.com).

Auth: X-Api-Key — resolved via services.pikzels_v2.resolve_public_api_key (PIKZELS_API_KEY preferred).
OpenAPI XOR pairs (image_url vs image_base64, etc.): we send **only** the non-empty side so
Pikzels never receives ``image_base64: \"\"`` (which fails their data-URL validator).

Upstream error envelope and codes: https://docs.pikzels.com/errors
Concurrency / 429: https://docs.pikzels.com/rate-limits
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Tuple

import httpx

from services.pikzels_v2 import PUBLIC_BASE, resolve_public_api_key
from services.pikzels_errors import format_pikzels_error_message

_log = logging.getLogger("uploadm8.pikzels_v2")
_PROCESS_LIMIT = max(1, int(os.environ.get("PIKZELS_MAX_CONCURRENCY", "8") or 8))
_REDIS_LIMIT = max(1, int(os.environ.get("PIKZELS_REDIS_MAX_CONCURRENCY", str(_PROCESS_LIMIT)) or _PROCESS_LIMIT))
_REDIS_WINDOW_MS = max(5_000, int(os.environ.get("PIKZELS_REDIS_SLOT_TTL_MS", "180000") or 180_000))
_process_semaphore = asyncio.Semaphore(_PROCESS_LIMIT)


def pikzels_api_key() -> str:
    """Same as resolve_public_api_key; kept for callers that import this name."""
    return resolve_public_api_key()


def pikzels_timeout_seconds() -> float:
    return float(os.environ.get("PIKZELS_API_TIMEOUT_SECONDS") or os.environ.get("PIKZELS_TIMEOUT_SECONDS", "120") or 120)


async def _try_redis_acquire(member: str) -> bool:
    """Best-effort cross-worker limiter. Falls back to process semaphore when Redis is absent."""
    try:
        import core.state

        rc = getattr(core.state, "redis_client", None)
        if rc is None:
            return True
        key = os.environ.get("PIKZELS_REDIS_LIMIT_KEY", "pikzels:v2:inflight")
        now = int(time.time() * 1000)
        script = """
        redis.call('ZREMRANGEBYSCORE', KEYS[1], 0, ARGV[1] - ARGV[2])
        local count = redis.call('ZCARD', KEYS[1])
        if count >= tonumber(ARGV[3]) then
          return 0
        end
        redis.call('ZADD', KEYS[1], ARGV[1], ARGV[4])
        redis.call('PEXPIRE', KEYS[1], ARGV[2])
        return 1
        """
        ok = await rc.eval(script, 1, key, now, _REDIS_WINDOW_MS, _REDIS_LIMIT, member)
        return bool(int(ok or 0))
    except Exception as e:
        _log.debug("pikzels redis limiter unavailable: %s", e)
        return True


async def _redis_release(member: str) -> None:
    try:
        import core.state

        rc = getattr(core.state, "redis_client", None)
        if rc is not None:
            await rc.zrem(os.environ.get("PIKZELS_REDIS_LIMIT_KEY", "pikzels:v2:inflight"), member)
    except Exception:
        pass


@asynccontextmanager
async def pikzels_rate_limit_slot():
    """
    Shared Pikzels limiter.

    Pikzels documents a default 10 in-flight requests per API key:
    https://docs.pikzels.com/rate-limits

    We cap each process below that and, when Redis is available, coordinate a
    best-effort cross-process slot counter as well.
    """
    member = f"{os.getpid()}:{uuid.uuid4().hex}"
    async with _process_semaphore:
        acquired = False
        deadline = time.monotonic() + float(os.environ.get("PIKZELS_LIMIT_WAIT_SECONDS", "30") or 30)
        while time.monotonic() < deadline:
            acquired = await _try_redis_acquire(member)
            if acquired:
                break
            await asyncio.sleep(0.15)
        if not acquired:
            raise httpx.HTTPStatusError(
                "Pikzels concurrency limit busy",
                request=httpx.Request("POST", PUBLIC_BASE),
                response=httpx.Response(429, json={"error": {"code": "RATE_LIMITED", "message": "Too many concurrent UploadM8 Pikzels requests"}}),
            )
        try:
            yield
        finally:
            await _redis_release(member)


def coerce_pikzels_v2_image_base64_fields(body: Dict[str, Any]) -> None:
    """
    Pikzels v2 validates ``image_base64`` / ``*_base64`` / ``image_base64s[]`` when present:
    non-empty values must be full ``data:image/*;base64,...`` URLs (not raw base64).

    Empty strings must be **omitted** — Pikzels treats ``\"\"`` as present and fails
    ("must start with data:image/") even when ``image_url`` is the active XOR side.

    Non-strings are dropped so JSON does not send ``null`` / numbers that trip the same validator.
    """
    for key in ("image_base64", "support_image_base64", "face_image_base64", "mask_base64"):
        if key not in body:
            continue
        v = body.get(key)
        if not isinstance(v, str):
            body.pop(key, None)
            continue
        t = v.strip()
        if not t:
            body.pop(key, None)
            continue
        low = t.lower()
        if low.startswith("data:image") and ";base64," in low:
            continue
        body[key] = f"data:image/jpeg;base64,{t}"[:14_000_000]
    arr = body.get("image_base64s")
    if isinstance(arr, list):
        if not arr:
            body.pop("image_base64s", None)
        else:
            out: List[str] = []
            for x in arr:
                if not isinstance(x, str):
                    continue
                t = x.strip()
                if not t:
                    continue
                low = t.lower()
                if low.startswith("data:image") and ";base64," in low:
                    out.append(t[:14_000_000])
                else:
                    out.append(f"data:image/jpeg;base64,{t}"[:14_000_000])
            body["image_base64s"] = out


def normalize_url_or_base64(body: Dict[str, Any], url_key: str, b64_key: str) -> None:
    """
    XOR url vs base64: keep **only** the non-empty side as a JSON key.

    Do not write ``\"\"`` for the unused side — Pikzels validates ``image_base64`` when the
    key exists and rejects empty strings even if ``image_url`` is set.
    """
    u = str(body.get(url_key) or "").strip() if body.get(url_key) is not None else ""
    b = str(body.get(b64_key) or "").strip() if body.get(b64_key) is not None else ""
    body.pop(url_key, None)
    body.pop(b64_key, None)
    if u and b:
        body[url_key] = u
    elif u:
        body[url_key] = u
    elif b:
        body[b64_key] = b


def resolve_pikzels_persona_style_xor(body: Dict[str, Any]) -> None:
    """
    Pikzels thumbnail endpoints do not combine ``persona`` and ``style`` pikzonalities.
    If both are set, keep ``persona`` (on-camera look) and omit ``style``.
    """
    if not isinstance(body, dict):
        return
    p = body.get("persona")
    s = body.get("style")
    p_ok = isinstance(p, str) and bool(p.strip())
    s_ok = isinstance(s, str) and bool(s.strip())
    if p_ok and s_ok:
        body.pop("style", None)
        _log.debug(
            "pikzels: dropped style (persona and style are mutually exclusive on thumbnail calls)"
        )


def trim_pikzonality_images(body: Dict[str, Any]) -> None:
    """
    Persona/style creation: exactly 3 refs from urls XOR base64s.

    Pikzels rejects bodies that include *both* keys (even with one empty array):
    "Provide either image_urls or image_base64s, not both".
    """
    raw_u = body.pop("image_urls", None)
    raw_b = body.pop("image_base64s", None)
    clean_u = [
        str(x).strip()
        for x in (raw_u if isinstance(raw_u, list) else [])
        if str(x).strip()
    ]
    clean_b = [
        str(x).strip()
        for x in (raw_b if isinstance(raw_b, list) else [])
        if str(x).strip()
    ]
    if len(clean_u) >= 3:
        body["image_urls"] = clean_u[:3]
    elif len(clean_b) >= 3:
        body["image_base64s"] = clean_b[:3]


async def pikzels_v2_post(path: str, json_body: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    key = pikzels_api_key()
    if not key:
        return 503, {
            "error": "missing_api_key",
            "message": "Set PIKZELS_API_KEY (preferred) or THUMB_RENDER_API_KEY for https://api.pikzels.com.",
        }
    if isinstance(json_body, dict):
        coerce_pikzels_v2_image_base64_fields(json_body)
        # Thumbnail create / recreate / prompt — same XOR rule as Pikzels UI.
        if "/v2/thumbnail/" in (path or ""):
            resolve_pikzels_persona_style_xor(json_body)
    url = f"{PUBLIC_BASE}{path}"
    headers = {"X-Api-Key": key, "Content-Type": "application/json"}
    timeout = pikzels_timeout_seconds()
    try:
        async with pikzels_rate_limit_slot():
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.post(url, headers=headers, json=json_body)
            data: Dict[str, Any]
            try:
                data = r.json() if r.content else {}
            except Exception:
                data = {"raw": (r.text or "")[:4000]}
            if r.status_code >= 400:
                _log.warning(
                    "pikzels POST %s HTTP %s: %s",
                    path,
                    r.status_code,
                    format_pikzels_error_message(data if isinstance(data, dict) else {}, max_len=400)
                    or str(data)[:300],
                )
            return r.status_code, data if isinstance(data, dict) else {"data": data}
    except httpx.TimeoutException:
        return 504, {"error": "timeout", "message": "Pikzels API request timed out."}
    except httpx.HTTPError as e:
        _log.warning("pikzels POST %s failed: %s", path, e)
        return 502, {"error": "upstream", "message": str(e)}


async def pikzels_v2_get(path: str) -> Tuple[int, Dict[str, Any]]:
    key = pikzels_api_key()
    if not key:
        return 503, {
            "error": "missing_api_key",
            "message": "Set PIKZELS_API_KEY (preferred) or THUMB_RENDER_API_KEY for https://api.pikzels.com.",
        }
    url = f"{PUBLIC_BASE}{path}"
    headers = {"X-Api-Key": key}
    timeout = pikzels_timeout_seconds()
    try:
        async with pikzels_rate_limit_slot():
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.get(url, headers=headers)
            try:
                data = r.json() if r.content else {}
            except Exception:
                data = {"raw": (r.text or "")[:4000]}
            return r.status_code, data if isinstance(data, dict) else {"data": data}
    except httpx.TimeoutException:
        return 504, {"error": "timeout", "message": "Pikzels API request timed out."}
    except httpx.HTTPError as e:
        _log.warning("pikzels GET %s failed: %s", path, e)
        return 502, {"error": "upstream", "message": str(e)}


async def pikzels_v2_patch(path: str, json_body: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    key = pikzels_api_key()
    if not key:
        return 503, {
            "error": "missing_api_key",
            "message": "Set PIKZELS_API_KEY (preferred) or THUMB_RENDER_API_KEY for https://api.pikzels.com.",
        }
    url = f"{PUBLIC_BASE}{path}"
    headers = {"X-Api-Key": key, "Content-Type": "application/json"}
    timeout = pikzels_timeout_seconds()
    try:
        async with pikzels_rate_limit_slot():
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.patch(url, headers=headers, json=json_body)
            try:
                data = r.json() if r.content else {}
            except Exception:
                data = {"raw": (r.text or "")[:4000]}
            if r.status_code >= 400:
                _log.warning(
                    "pikzels PATCH %s HTTP %s: %s",
                    path,
                    r.status_code,
                    format_pikzels_error_message(data if isinstance(data, dict) else {}, max_len=400)
                    or str(data)[:300],
                )
            return r.status_code, data if isinstance(data, dict) else {"data": data}
    except httpx.TimeoutException:
        return 504, {"error": {"code": "GENERATION_TIMEOUT", "message": "Pikzels API patch request timed out."}}
    except httpx.HTTPError as e:
        _log.warning("pikzels PATCH %s failed: %s", path, e)
        return 502, {"error": {"code": "UPSTREAM", "message": str(e)}}


async def pikzels_v2_delete(path: str) -> Tuple[int, Dict[str, Any]]:
    key = pikzels_api_key()
    if not key:
        return 503, {
            "error": "missing_api_key",
            "message": "Set PIKZELS_API_KEY (preferred) or THUMB_RENDER_API_KEY for https://api.pikzels.com.",
        }
    url = f"{PUBLIC_BASE}{path}"
    headers = {"X-Api-Key": key}
    timeout = pikzels_timeout_seconds()
    try:
        async with pikzels_rate_limit_slot():
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.delete(url, headers=headers)
            try:
                data = r.json() if r.content else {}
            except Exception:
                data = {"raw": (r.text or "")[:4000]}
            if r.status_code >= 400:
                _log.warning(
                    "pikzels DELETE %s HTTP %s: %s",
                    path,
                    r.status_code,
                    format_pikzels_error_message(data if isinstance(data, dict) else {}, max_len=400)
                    or str(data)[:300],
                )
            return r.status_code, data if isinstance(data, dict) else {"data": data}
    except httpx.TimeoutException:
        return 504, {"error": {"code": "GENERATION_TIMEOUT", "message": "Pikzels API delete request timed out."}}
    except httpx.HTTPError as e:
        _log.warning("pikzels DELETE %s failed: %s", path, e)
        return 502, {"error": {"code": "UPSTREAM", "message": str(e)}}
