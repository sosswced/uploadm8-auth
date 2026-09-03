"""
Meta (Facebook / Instagram) OAuth connect helpers — multi-destination pick.

After Meta Login, users who manage multiple Pages (or multiple IG Professional
accounts) choose which destination to store. Tokens stay in Redis until pick.
"""
from __future__ import annotations

import json
import logging
import secrets
from typing import Any, Dict, List, Optional, Tuple

import httpx

import core.state
from services.meta_oauth import META_GRAPH_API_VERSION, fetch_managed_pages

logger = logging.getLogger("uploadm8-api")

_META_PICK_TTL = 600  # 10 minutes — same order as oauth_state


async def store_meta_pick_pending(data: dict) -> str:
    """Persist pending Meta connect payload; returns opaque pick token."""
    if not core.state.redis_client:
        raise RuntimeError("OAuth temporarily unavailable (cache offline)")
    token = secrets.token_urlsafe(32)
    await core.state.redis_client.setex(
        f"oauth_meta_pick:{token}",
        _META_PICK_TTL,
        json.dumps(data),
    )
    return token


async def pop_meta_pick_pending(token: str) -> Optional[dict]:
    if not core.state.redis_client or not token:
        return None
    key = f"oauth_meta_pick:{token}"
    raw = await core.state.redis_client.get(key)
    if not raw:
        return None
    await core.state.redis_client.delete(key)
    try:
        return json.loads(raw)
    except Exception:
        return None


async def list_facebook_page_destinations(
    client: httpx.AsyncClient,
    user_access_token: str,
) -> List[Dict[str, Any]]:
    """Return Pages the user can publish to (id, name, username, avatar, access_token)."""
    pages = await fetch_managed_pages(
        client,
        user_access_token,
        fields="id,name,username,access_token,picture",
    )
    out: List[Dict[str, Any]] = []
    for page in pages or []:
        page_id = str(page.get("id") or "").strip()
        page_token = str(page.get("access_token") or "").strip()
        if not page_id or not page_token:
            continue
        pic = page.get("picture") or {}
        if isinstance(pic, dict):
            avatar = str((pic.get("data") or {}).get("url") or "").strip()
        else:
            avatar = ""
        out.append(
            {
                "destination_id": page_id,
                "page_id": page_id,
                "name": str(page.get("name") or "Facebook Page").strip() or "Facebook Page",
                "username": (str(page.get("username") or "").strip()),
                "avatar": avatar,
                "access_token": page_token,
                "label": str(page.get("name") or page_id),
            }
        )
    return out


async def list_instagram_destinations(
    client: httpx.AsyncClient,
    user_access_token: str,
) -> List[Dict[str, Any]]:
    """Return IG Professional accounts linked to managed Pages."""
    pages = await fetch_managed_pages(
        client,
        user_access_token,
        fields="id,name,access_token",
    )
    out: List[Dict[str, Any]] = []
    for page in pages or []:
        page_id = str(page.get("id") or "").strip()
        page_token = str(page.get("access_token") or "").strip()
        if not page_id or not page_token:
            continue
        try:
            ig_response = await client.get(
                f"https://graph.facebook.com/{META_GRAPH_API_VERSION}/{page_id}",
                params={
                    "fields": "instagram_business_account",
                    "access_token": page_token,
                },
            )
            ig_data = ig_response.json() if ig_response.content else {}
        except Exception as e:
            logger.debug("IG link check failed for page %s: %s", page_id, e)
            continue
        ig_ba = ig_data.get("instagram_business_account") if isinstance(ig_data, dict) else None
        if not isinstance(ig_ba, dict):
            continue
        ig_account_id = str(ig_ba.get("id") or "").strip()
        if not ig_account_id:
            continue
        try:
            ig_details_response = await client.get(
                f"https://graph.facebook.com/{META_GRAPH_API_VERSION}/{ig_account_id}",
                params={
                    "fields": "id,username,name,profile_picture_url",
                    "access_token": page_token,
                },
            )
            if ig_details_response.status_code == 200:
                instagram_account = ig_details_response.json() or {}
            else:
                logger.warning(
                    "Instagram profile fetch HTTP %s (degraded): %s",
                    ig_details_response.status_code,
                    (ig_details_response.text or "")[:240],
                )
                instagram_account = {
                    "id": ig_account_id,
                    "username": "",
                    "name": f"Instagram account {ig_account_id}",
                    "profile_picture_url": "",
                }
        except Exception as e:
            logger.warning("Instagram profile fetch failed: %s", e)
            instagram_account = {
                "id": ig_account_id,
                "username": "",
                "name": f"Instagram account {ig_account_id}",
                "profile_picture_url": "",
            }
        uname = str(instagram_account.get("username") or "").strip()
        name = (
            str(instagram_account.get("name") or "").strip()
            or uname
            or f"Instagram {ig_account_id}"
        )
        page_name = str(page.get("name") or "").strip()
        label = f"@{uname}" if uname else name
        if page_name:
            label = f"{label} · {page_name}"
        out.append(
            {
                "destination_id": ig_account_id,
                "ig_user_id": ig_account_id,
                "page_id": page_id,
                "page_name": page_name,
                "name": name,
                "username": uname,
                "avatar": str(instagram_account.get("profile_picture_url") or "").strip(),
                "access_token": page_token,
                "label": label,
            }
        )
    return out


def pick_destination(
    destinations: List[Dict[str, Any]],
    *,
    expected_provider_id: Optional[str] = None,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Auto-resolve a destination when possible.

    Returns (destination_or_None, reason) where reason is:
      'none' | 'auto_single' | 'auto_reconnect' | 'need_pick'
    """
    if not destinations:
        return None, "none"
    want = str(expected_provider_id or "").strip()
    if want:
        for d in destinations:
            if str(d.get("destination_id") or "") == want:
                return d, "auto_reconnect"
            if str(d.get("page_id") or "") == want:
                return d, "auto_reconnect"
            if str(d.get("ig_user_id") or "") == want:
                return d, "auto_reconnect"
        return None, "need_pick"
    if len(destinations) == 1:
        return destinations[0], "auto_single"
    return None, "need_pick"


def meta_pick_html(
    *,
    platform: str,
    pick_token: str,
    destinations: List[Dict[str, Any]],
    post_target: str,
    api_base: str,
) -> str:
    """Self-contained picker page shown inside the OAuth popup."""
    plat = "Instagram" if platform == "instagram" else "Facebook Page"
    rows = []
    for d in destinations:
        dest_id = str(d.get("destination_id") or "")
        label = str(d.get("label") or d.get("name") or dest_id)
        avatar = str(d.get("avatar") or "")
        avatar_html = (
            f'<img src="{_esc_attr(avatar)}" alt="" '
            f'style="width:40px;height:40px;border-radius:50%;object-fit:cover;" />'
            if avatar.startswith("http")
            else '<div style="width:40px;height:40px;border-radius:50%;background:#333;"></div>'
        )
        rows.append(
            f"""
            <button type="submit" name="destination_id" value="{_esc_attr(dest_id)}"
              style="display:flex;align-items:center;gap:12px;width:100%;padding:12px 14px;
                     margin:0 0 10px;border:1px solid rgba(255,255,255,0.12);border-radius:10px;
                     background:rgba(255,255,255,0.04);color:#fff;cursor:pointer;text-align:left;
                     font:inherit;">
              {avatar_html}
              <span style="flex:1;font-weight:600;">{_esc(label)}</span>
            </button>
            """
        )
    form_action = f"{api_base.rstrip('/')}/api/oauth/{platform}/select-destination"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Choose {plat}</title>
</head>
<body style="font-family:system-ui,sans-serif;margin:0;min-height:100vh;background:#0a0a0f;color:#fff;
             display:flex;align-items:center;justify-content:center;padding:24px;">
  <div style="max-width:420px;width:100%;">
    <h1 style="font-size:1.25rem;margin:0 0 8px;">Choose a {plat}</h1>
    <p style="color:#9ca3af;font-size:0.9rem;margin:0 0 20px;">
      You manage more than one. Pick which account UploadM8 should connect for publishing and analytics.
    </p>
    <form method="POST" action="{_esc_attr(form_action)}">
      <input type="hidden" name="pick_token" value="{_esc_attr(pick_token)}"/>
      <input type="hidden" name="parent_origin" value="{_esc_attr(post_target)}"/>
      {''.join(rows)}
    </form>
    <p style="color:#6b7280;font-size:0.75rem;margin-top:16px;">
      You can connect additional Pages or Instagram accounts later from Connected Accounts.
    </p>
  </div>
</body>
</html>
"""


async def resolve_or_pick_destination(
    *,
    platform: str,
    destinations: List[Dict[str, Any]],
    expected_provider_id: Optional[str],
    pending_fields: dict,
    post_target: str,
    api_base: str,
):
    """
    Auto-pick a destination or return picker HTMLResponse.

    Returns (chosen_dict_or_None, html_response_or_None).
    When html_response is set, the OAuth callback should return it immediately.
    """
    from fastapi.responses import HTMLResponse

    chosen, reason = pick_destination(
        destinations, expected_provider_id=expected_provider_id
    )
    if reason == "need_pick":
        pick_token = await store_meta_pick_pending(
            {
                **pending_fields,
                "platform": platform,
                "parent_origin": post_target,
                "destinations": destinations,
                "reconnect_expected_provider_account_id": expected_provider_id,
            }
        )
        return None, HTMLResponse(
            meta_pick_html(
                platform=platform,
                pick_token=pick_token,
                destinations=destinations,
                post_target=post_target,
                api_base=api_base,
            )
        )
    return chosen, None


def _esc(s: str) -> str:
    return (
        str(s or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _esc_attr(s: str) -> str:
    return _esc(s).replace("'", "&#39;")
