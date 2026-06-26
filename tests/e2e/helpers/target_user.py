"""Canonical E2E target user for admin / wallet / settings feature sweeps."""

from __future__ import annotations

from typing import Any, Callable

import httpx

from tests.e2e.helpers.config import (
    e2e_target_user_id,
    e2e_target_user_name,
    e2e_target_user_search_terms,
)


class TargetUserNotFound(RuntimeError):
    pass


def target_user_id() -> str:
    return e2e_target_user_id()


def target_user_name() -> str:
    return e2e_target_user_name()


def resolve_target_user(client: httpx.Client) -> dict[str, Any]:
    """Load target user by ID; fall back to name search."""
    uid = target_user_id()
    wallet = client.get(f"/api/admin/users/{uid}/wallet")
    if wallet.status_code == 200:
        body = wallet.json()
        user = body.get("user") if isinstance(body, dict) else None
        if isinstance(user, dict) and str(user.get("id")) == uid:
            return user
    for term in e2e_target_user_search_terms():
        r = client.get("/api/admin/users", params={"search": term, "limit": 50})
        r.raise_for_status()
        for row in r.json().get("users") or []:
            if str(row.get("id")) == uid:
                return row
    r = client.get("/api/admin/users", params={"limit": 2000})
    r.raise_for_status()
    for row in r.json().get("users") or []:
        if str(row.get("id")) == uid:
            return row
    raise TargetUserNotFound(
        f"Target user {uid!r} ({target_user_name()!r}) not found — "
        "set E2E_TARGET_USER_ID / E2E_TARGET_USER_NAME in .env"
    )


def _check_search_has_user(resp: httpx.Response, uid: str) -> str:
    rows = resp.json().get("users") or []
    if any(str(r.get("id")) == uid for r in rows):
        return "found in search results"
    return f"user {uid} missing from search ({len(rows)} rows)"


def _check_wallet_user(resp: httpx.Response, uid: str) -> str:
    body = resp.json()
    user = body.get("user") or {}
    if str(user.get("id")) != uid:
        return f"wallet user id mismatch: {user.get('id')}"
    if "wallet" not in body:
        return "wallet payload missing"
    return "wallet ok"


def _check_audit_user(resp: httpx.Response, uid: str) -> str:
    items = resp.json().get("items") or resp.json().get("events") or []
    return f"{len(items)} audit event(s)"


def _check_leaderboard_has_user(resp: httpx.Response, uid: str) -> str:
    body = resp.json()
    if isinstance(body, list):
        rows = body
    elif isinstance(body, dict):
        rows = body.get("users") or body.get("items") or []
    else:
        return "leaderboard shape ok"
    if not isinstance(rows, list):
        return "leaderboard shape ok"
    if any(str(r.get("id")) == uid for r in rows):
        return "user on leaderboard"
    return "user not in top-10 leaderboard (ok if low activity)"


def target_user_admin_api_checks(uid: str | None = None) -> list[dict[str, Any]]:
    """
    Read-only admin API matrix for the target user.

    Each entry: id, method, path, params?, assert_fn(resp) -> detail str (empty = pass).
    """
    uid = uid or target_user_id()
    name_term = target_user_name().split()[0] if target_user_name() else "Johnny"

    def _ok(_: httpx.Response) -> str:
        return ""

    checks: list[dict[str, Any]] = [
        {
            "id": "users.search_name",
            "method": "GET",
            "path": "/api/admin/users",
            "params": {"search": name_term, "limit": 50},
            "assert_fn": lambda r: _check_search_has_user(r, uid),
        },
        {
            "id": "users.search_id_prefix",
            "method": "GET",
            "path": "/api/admin/users",
            "params": {"search": uid.split("-")[0], "limit": 50},
            "assert_fn": lambda r: _check_search_has_user(r, uid),
        },
        {
            "id": "users.wallet",
            "method": "GET",
            "path": f"/api/admin/users/{uid}/wallet",
            "assert_fn": lambda r: _check_wallet_user(r, uid),
        },
        {
            "id": "audit.user",
            "method": "GET",
            "path": "/api/admin/audit",
            "params": {"user_id": uid, "limit": 50, "source": "all"},
            "assert_fn": lambda r: _check_audit_user(r, uid),
        },
        {
            "id": "audit.user_admin_only",
            "method": "GET",
            "path": "/api/admin/audit",
            "params": {"user_id": uid, "limit": 25, "source": "admin"},
            "assert_fn": lambda r: _check_audit_user(r, uid),
        },
        {
            "id": "audit.user_system",
            "method": "GET",
            "path": "/api/admin/audit",
            "params": {"user_id": uid, "limit": 25, "source": "system"},
            "assert_fn": lambda r: _check_audit_user(r, uid),
        },
        {
            "id": "wallet_disputes",
            "method": "GET",
            "path": "/api/admin/wallet-disputes",
            "params": {"limit": 100},
            "assert_fn": _ok,
        },
        {
            "id": "analytics.users",
            "method": "GET",
            "path": "/api/admin/analytics/users",
            "assert_fn": _ok,
        },
        {
            "id": "analytics.overview",
            "method": "GET",
            "path": "/api/admin/analytics/overview",
            "params": {"days": 30},
            "assert_fn": _ok,
        },
        {
            "id": "leaderboard.uploads",
            "method": "GET",
            "path": "/api/admin/leaderboard",
            "params": {"range": "30d", "sort": "uploads"},
            "assert_fn": lambda r: _check_leaderboard_has_user(r, uid),
            "soft": True,
        },
        {
            "id": "top_users",
            "method": "GET",
            "path": "/api/admin/top-users",
            "params": {"limit": 20},
            "assert_fn": lambda r: _check_leaderboard_has_user(r, uid),
            "soft": True,
        },
        {
            "id": "settings",
            "method": "GET",
            "path": "/api/admin/settings",
            "assert_fn": _ok,
        },
        {
            "id": "notification_settings",
            "method": "GET",
            "path": "/api/admin/notification-settings",
            "assert_fn": _ok,
        },
        {
            "id": "billing.catalog",
            "method": "GET",
            "path": "/api/admin/billing/catalog",
            "assert_fn": _ok,
        },
        {
            "id": "billing.effective_tiers",
            "method": "GET",
            "path": "/api/admin/billing/effective-tiers",
            "assert_fn": _ok,
        },
    ]
    return checks


def run_target_user_api_checks(client: httpx.Client, uid: str | None = None) -> list[dict[str, Any]]:
    """Execute API checks; returns per-check result dicts."""
    uid = uid or target_user_id()
    resolve_target_user(client)
    results: list[dict[str, Any]] = []
    for spec in target_user_admin_api_checks(uid):
        t0_detail = ""
        status = "PASS"
        try:
            kwargs: dict[str, Any] = {}
            if spec.get("params"):
                kwargs["params"] = spec["params"]
            fn: Callable[[httpx.Response], str] = spec["assert_fn"]
            resp = client.request(spec["method"], spec["path"], **kwargs)
            if resp.status_code >= 400:
                status = "FAIL"
                t0_detail = f"HTTP {resp.status_code}: {resp.text[:200]}"
            else:
                msg = fn(resp)
                if msg and ("missing" in msg.lower() or "mismatch" in msg.lower()):
                    status = "FAIL" if not spec.get("soft") else "PASS"
                    t0_detail = msg
                else:
                    t0_detail = msg or "ok"
        except Exception as e:
            status = "FAIL"
            t0_detail = str(e)[:300]
        results.append(
            {
                "id": spec["id"],
                "status": status,
                "detail": t0_detail,
                "path": spec["path"],
            }
        )
    return results
