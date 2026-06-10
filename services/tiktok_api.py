"""
TikTok Open API v2 — URL and response envelope rules (single source for app, worker, jobs).

Docs: video list / video query use ?fields= in the query string, not the JSON body.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Mapping, Optional, Tuple
from urllib.parse import quote

import httpx

logger = logging.getLogger(__name__)

TIKTOK_USER_INFO_URL = "https://open.tiktokapis.com/v2/user/info/"
# Scope tiers (https://developers.tiktok.com/bulletin/user-info-scope-migration):
# - user.info.basic: open_id, display_name, avatar_*
# - user.info.profile: username, bio_description, is_verified, profile_deep_link
# - user.info.stats: follower_count, following_count, likes_count, video_count
TIKTOK_USER_INFO_FIELDS_BASIC = (
    "open_id,union_id,display_name,avatar_url,avatar_url_100,avatar_large_url"
)
TIKTOK_USER_INFO_FIELDS_PROFILE = (
    "open_id,union_id,username,display_name,avatar_url,avatar_url_100,avatar_large_url,"
    "bio_description,is_verified,profile_deep_link"
)
TIKTOK_USER_INFO_FIELDS_STATS = "follower_count,following_count,likes_count,video_count"
# Legacy alias — do not request username/stats here (breaks basic-only tokens).
TIKTOK_USER_INFO_FIELDS_FULL = TIKTOK_USER_INFO_FIELDS_PROFILE

# Single list used by catalog sync and any caller of ``tiktok_video_list_url`` — no short/long split;
# all published videos are requested with the same field set.
TIKTOK_VIDEO_LIST_FIELDS = (
    "id,title,video_description,duration,cover_image_url,share_url,"
    "view_count,like_count,comment_count,share_count,create_time"
)
TIKTOK_VIDEO_QUERY_FIELDS = "id,view_count,like_count,comment_count,share_count"

TIKTOK_CREATOR_INFO_URL = "https://open.tiktokapis.com/v2/post/publish/creator_info/query/"

TIKTOK_PRIVACY_LABELS: dict[str, str] = {
    "PUBLIC_TO_EVERYONE": "Everyone",
    "MUTUAL_FOLLOW_FRIENDS": "Friends",
    "FOLLOWER_OF_CREATOR": "Followers",
    "SELF_ONLY": "Only me",
}

TIKTOK_KNOWN_PRIVACY_LEVELS = frozenset(TIKTOK_PRIVACY_LABELS.keys())

TIKTOK_MUSIC_USAGE_CONFIRMATION_URL = (
    "https://www.tiktok.com/legal/page/global/music-usage-confirmation/en"
)
TIKTOK_BRANDED_CONTENT_POLICY_URL = "https://www.tiktok.com/legal/page/global/bc-policy/en"


def tiktok_app_audited() -> bool:
    """True after TikTok Content Posting API audit passes. Set ``TIKTOK_APP_AUDITED=1``."""
    v = (os.environ.get("TIKTOK_APP_AUDITED") or "").strip().lower()
    return v in ("1", "true", "yes", "on")


def tiktok_unaudited_mode() -> bool:
    """Apps pending TikTok audit must show private-only UX and clamp posts to SELF_ONLY."""
    return not tiktok_app_audited()


def tiktok_force_private_unaudited() -> bool:
    """Clamp Direct Post privacy to SELF_ONLY while unaudited (or when force flag set)."""
    if tiktok_unaudited_mode():
        return True
    v = (os.environ.get("TIKTOK_FORCE_PRIVATE_UNAUDITED") or "").strip().lower()
    return v in ("1", "true", "yes", "on")


def tiktok_video_list_url() -> str:
    return (
        "https://open.tiktokapis.com/v2/video/list/"
        f"?fields={quote(TIKTOK_VIDEO_LIST_FIELDS)}"
    )


def tiktok_video_query_url() -> str:
    return (
        "https://open.tiktokapis.com/v2/video/query/"
        f"?fields={quote(TIKTOK_VIDEO_QUERY_FIELDS)}"
    )


def tiktok_envelope_error(body: Any) -> Optional[str]:
    """TikTok often returns HTTP 200 with error.code != 'ok' in JSON."""
    if not isinstance(body, dict):
        return None
    err = body.get("error")
    if not isinstance(err, dict):
        return None
    code = err.get("code")
    if code is None or str(code).lower() == "ok":
        return None
    return (err.get("message") or str(code) or "tiktok_api_error")[:500]


def tiktok_parse_oauth_token_response(resp_json: Any) -> Tuple[str, str, Optional[str], dict]:
    """
    Normalize TikTok /v2/oauth/token/ JSON (flat or wrapped under ``data``).
    Returns (access_token, open_id, refresh_token_or_none, merged_dict_for_blob).
    """
    if not isinstance(resp_json, dict):
        return "", "", None, {}
    nested = resp_json.get("data")
    src: dict = nested if isinstance(nested, dict) else resp_json
    access = str(src.get("access_token") or resp_json.get("access_token") or "").strip()
    open_id = str(src.get("open_id") or resp_json.get("open_id") or "").strip()
    refresh = src.get("refresh_token")
    if refresh is None:
        refresh = resp_json.get("refresh_token")
    rft: Optional[str]
    if refresh is None:
        rft = None
    elif isinstance(refresh, str):
        rft = refresh
    else:
        rft = str(refresh)
    return access, open_id, rft, src


def tiktok_extract_user_from_info_body(body: Any) -> dict:
    """Pull the user object from /v2/user/info/ response variants."""
    if not isinstance(body, dict):
        return {}
    data = body.get("data")
    if isinstance(data, dict):
        user = data.get("user")
        if isinstance(user, dict):
            return user
        if any(k in data for k in ("open_id", "display_name", "avatar_url", "avatar_url_100")):
            return data
    user = body.get("user")
    if isinstance(user, dict):
        return user
    return {}


def tiktok_identity_from_user_object(user_obj: Mapping[str, Any]) -> dict:
    """
    Map TikTok user JSON to platform_tokens-style fields.
    Returns keys: account_id, account_name, account_username, account_avatar (may be "").
    """
    if not isinstance(user_obj, Mapping):
        return {
            "account_id": "",
            "account_name": "",
            "account_username": "",
            "account_avatar": "",
        }
    incoming_open_id = (
        str(user_obj.get("open_id") or "").strip()
        or str(user_obj.get("user_id") or "").strip()
    )
    display_name = (
        str(user_obj.get("display_name") or "").strip()
        or str(user_obj.get("name") or "").strip()
        or str(user_obj.get("nickname") or "").strip()
    )
    username = (
        str(user_obj.get("username") or "").strip()
        or str(user_obj.get("unique_id") or "").strip()
        or str(user_obj.get("username_normalized") or "").strip()
    )
    avatar_url = (
        str(user_obj.get("avatar_large_url") or "").strip()
        or str(user_obj.get("avatar_url_100") or "").strip()
        or str(user_obj.get("avatar_url_hd") or "").strip()
        or str(user_obj.get("avatar_url") or "").strip()
        or str(user_obj.get("profile_image") or "").strip()
        or str(user_obj.get("avatar") or "").strip()
    )
    return {
        "account_id": incoming_open_id,
        "account_name": display_name,
        "account_username": username,
        "account_avatar": avatar_url,
    }


def _merge_tiktok_user_objects(base: Mapping[str, Any], extra: Mapping[str, Any]) -> dict:
    out = dict(base) if isinstance(base, Mapping) else {}
    if not isinstance(extra, Mapping):
        return out
    for key, val in extra.items():
        if val is None:
            continue
        if isinstance(val, str) and not val.strip():
            continue
        if key not in out or not out.get(key):
            out[key] = val
    return out


async def _tiktok_user_info_get(
    client: httpx.AsyncClient,
    access_token: str,
    fields: str,
) -> Tuple[Optional[dict], str]:
    """Single GET /v2/user/info/; returns (user_obj or None, log_hint)."""
    headers = {
        "Authorization": f"Bearer {access_token.strip()}",
        "Accept": "application/json",
    }
    try:
        resp = await client.get(
            TIKTOK_USER_INFO_URL,
            params={"fields": fields},
            headers=headers,
        )
        try:
            body = resp.json() if resp.content else {}
        except Exception:
            body = {}
        env_err = tiktok_envelope_error(body)
        if resp.status_code != 200 or env_err:
            return None, (
                f"GET fields={fields[:48]} HTTP {resp.status_code} envelope={env_err!r}"
            )
        user_obj = tiktok_extract_user_from_info_body(body)
        if user_obj:
            return user_obj, "ok"
        return None, f"GET fields={fields[:48]} empty user payload"
    except Exception as e:
        return None, f"GET fields={fields[:48]} exception: {type(e).__name__}: {e}"


async def fetch_tiktok_user_profile_for_oauth(
    client: httpx.AsyncClient,
    access_token: str,
) -> dict:
    """
    Fetch display name + avatar after OAuth using ``user.info.basic`` fields only.

    TikTok scope migration: ``open_id``, ``display_name``, and ``avatar_*`` live under
    ``user.info.basic``. The old code incorrectly included ``username`` in the basic
    field list; ``username`` requires ``user.info.profile`` and caused the whole
    ``/v2/user/info/`` call to fail — empty name/avatar and ui-avatars placeholders.
    """
    if not access_token or not str(access_token).strip():
        return tiktok_identity_from_user_object({})

    user_obj, hint = await _tiktok_user_info_get(
        client, access_token, TIKTOK_USER_INFO_FIELDS_BASIC
    )
    if user_obj:
        ident = tiktok_identity_from_user_object(user_obj)
        logger.info(
            "TikTok user/info (basic) has_name=%s has_avatar=%s open_id=%s",
            bool(ident.get("account_name")),
            bool(ident.get("account_avatar")),
            bool(ident.get("account_id")),
        )
        return ident

    logger.warning("TikTok user/info (basic) failed: %s", hint)
    return tiktok_identity_from_user_object({})


def normalize_tiktok_creator_info(body: Any) -> dict:
    """Normalize /v2/post/publish/creator_info/query/ response for the export UI."""
    if not isinstance(body, dict):
        return {}
    data = body.get("data")
    if not isinstance(data, dict):
        return {}
    privacy_opts = data.get("privacy_level_options") or []
    if not isinstance(privacy_opts, list):
        privacy_opts = []
    privacy_opts = [str(x).strip() for x in privacy_opts if str(x).strip()]
    max_dur = data.get("max_video_post_duration_sec")
    try:
        max_dur = int(max_dur) if max_dur is not None else None
    except (TypeError, ValueError):
        max_dur = None
    return {
        "creator_avatar_url": str(data.get("creator_avatar_url") or "").strip(),
        "creator_username": str(data.get("creator_username") or "").strip(),
        "creator_nickname": str(data.get("creator_nickname") or "").strip(),
        "privacy_level_options": privacy_opts,
        "privacy_level_labels": {
            code: TIKTOK_PRIVACY_LABELS.get(code, code.replace("_", " ").title())
            for code in privacy_opts
        },
        "comment_disabled": bool(data.get("comment_disabled")),
        "duet_disabled": bool(data.get("duet_disabled")),
        "stitch_disabled": bool(data.get("stitch_disabled")),
        "max_video_post_duration_sec": max_dur,
    }


async def fetch_tiktok_creator_info(
    client: httpx.AsyncClient,
    access_token: str,
) -> tuple[dict, Optional[str]]:
    """
    Query creator info required before rendering TikTok export UI.
    Returns (normalized_creator_info, error_message_or_none).
    """
    if not access_token or not str(access_token).strip():
        return {}, "Missing TikTok access token"
    headers = {
        "Authorization": f"Bearer {access_token.strip()}",
        "Content-Type": "application/json; charset=UTF-8",
    }
    try:
        resp = await client.post(TIKTOK_CREATOR_INFO_URL, headers=headers, json={})
        try:
            body = resp.json() if resp.content else {}
        except Exception:
            body = {}
        env_err = tiktok_envelope_error(body)
        if resp.status_code != 200 or env_err:
            hint = env_err or (resp.text or "")[:300]
            return {}, f"TikTok creator_info failed ({resp.status_code}): {hint}"
        info = normalize_tiktok_creator_info(body)
        if not info.get("privacy_level_options"):
            return {}, "TikTok creator_info returned no privacy_level_options"
        return info, None
    except Exception as e:
        return {}, f"TikTok creator_info exception: {type(e).__name__}: {e}"


def _coerce_bool(val: Any, default: bool = False) -> bool:
    if isinstance(val, bool):
        return val
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return bool(val)
    s = str(val).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off", ""):
        return False
    return default


def normalize_tiktok_post_settings(raw: Any) -> dict:
    """
    Normalize client payload for one TikTok account export form.
    Keys match TikTok Direct Post post_info mapping in publish_stage.
    """
    if not isinstance(raw, dict):
        return {}
    allowed_raw = raw.get("allowed_privacy_levels") or raw.get("privacy_level_options") or []
    allowed: list[str] = []
    if isinstance(allowed_raw, (list, tuple)):
        allowed = [str(x).strip() for x in allowed_raw if str(x).strip()]
    max_dur = raw.get("max_video_post_duration_sec")
    try:
        max_dur_int = int(max_dur) if max_dur is not None else None
    except (TypeError, ValueError):
        max_dur_int = None
    out = {
        "privacy_level": str(raw.get("privacy_level") or "").strip(),
        "allowed_privacy_levels": allowed,
        "max_video_post_duration_sec": max_dur_int,
        "allow_comment": _coerce_bool(raw.get("allow_comment"), False),
        "allow_duet": _coerce_bool(raw.get("allow_duet"), False),
        "allow_stitch": _coerce_bool(raw.get("allow_stitch"), False),
        "commercial_disclosure_enabled": _coerce_bool(
            raw.get("commercial_disclosure_enabled"), False
        ),
        "brand_organic": _coerce_bool(raw.get("brand_organic"), False),
        "brand_content": _coerce_bool(raw.get("brand_content"), False),
        "title": str(raw.get("title") or "").strip(),
        "user_consent": _coerce_bool(raw.get("user_consent"), False),
    }
    if not out["commercial_disclosure_enabled"]:
        out["brand_organic"] = False
        out["brand_content"] = False
    return out


def validate_tiktok_post_settings(settings: Mapping[str, Any]) -> list[str]:
    """Return human-readable validation errors (empty list = OK)."""
    s = normalize_tiktok_post_settings(settings)
    errors: list[str] = []
    if not s.get("privacy_level"):
        errors.append("Select who can view this TikTok post.")
    elif s["privacy_level"] not in TIKTOK_KNOWN_PRIVACY_LEVELS:
        errors.append("Invalid TikTok privacy level.")
    else:
        allowed = s.get("allowed_privacy_levels") or []
        if allowed and s["privacy_level"] not in allowed:
            errors.append("Selected privacy level is not available for this TikTok account.")
    if not s.get("user_consent"):
        errors.append("Confirm TikTok posting consent before publishing.")
    if s.get("commercial_disclosure_enabled"):
        if not s.get("brand_organic") and not s.get("brand_content"):
            errors.append(
                "You need to indicate if your content promotes yourself, a third party, or both."
            )
        if s.get("brand_content") and s.get("privacy_level") == "SELF_ONLY":
            errors.append("Branded content visibility cannot be set to private.")
    return errors


def resolve_tiktok_post_settings_for_account(
    raw_settings: Any,
    account_id: str,
) -> Optional[dict]:
    """Look up per-account TikTok export settings saved on the upload."""
    if not isinstance(raw_settings, dict):
        return None
    by_account = raw_settings.get("by_account")
    if isinstance(by_account, dict):
        entry = by_account.get(str(account_id))
        if isinstance(entry, dict):
            return normalize_tiktok_post_settings(entry)
    if raw_settings.get("privacy_level"):
        return normalize_tiktok_post_settings(raw_settings)
    return None


def tiktok_post_info_from_settings(settings: Mapping[str, Any], *, title: str) -> dict:
    """Map export UI settings to TikTok Direct Post ``post_info`` fields."""
    s = normalize_tiktok_post_settings(settings)
    post_info: dict[str, Any] = {
        "title": title,
        "privacy_level": s["privacy_level"],
        "disable_comment": not s["allow_comment"],
        "disable_duet": not s["allow_duet"],
        "disable_stitch": not s["allow_stitch"],
    }
    if s["commercial_disclosure_enabled"]:
        post_info["brand_organic_toggle"] = bool(s["brand_organic"])
        post_info["brand_content_toggle"] = bool(s["brand_content"])
    return post_info
