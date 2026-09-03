"""
Meta (Facebook / Instagram) OAuth scope selection and permission checks.

Use META_OAUTH_MODE=minimal only for a restricted reviewer demo. Production uses
full scopes now that Meta has approved publishing and insights.

Environment:
  META_OAUTH_MODE   full | minimal | custom (default: full)
  META_INSTAGRAM_OAUTH_SCOPE   optional override when META_OAUTH_MODE=custom
  META_FACEBOOK_OAUTH_SCOPE    optional override when META_OAUTH_MODE=custom
"""

from __future__ import annotations

import contextvars
import json
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Mapping, Optional

import httpx

# Approved for your app (demo-safe): list Pages, Business Manager linkage, Page engagement.
_SCOPES_MINIMAL = "pages_show_list,pages_read_engagement,business_management"

# Production: publishing + insights + listing (requires Meta App Review per permission).
_SCOPES_INSTAGRAM_FULL = (
    "instagram_basic,"
    "instagram_content_publish,"
    "instagram_manage_insights,"
    "pages_show_list,"
    "pages_read_engagement,"
    "pages_read_user_content,"
    "business_management"
)

# Production: publishing + insights + listing (Meta App Review approved).
# Do not request publish_video — Meta defines that as LIVE streaming only.
_SCOPES_FACEBOOK_FULL = (
    "pages_manage_posts,"
    "pages_read_engagement,"
    "pages_read_user_content,"
    "pages_show_list,"
    "read_insights"
)

# Official Meta Allowed Usage (developers.facebook.com/docs/permissions) + how UploadM8 stays inside it.
# Paste `app_review_notes` into App Review. Do not claim live streaming, ads, or other Pages.
META_PERMISSION_ALLOWED_USAGE: Dict[str, Dict[str, str]] = {
    "pages_manage_posts": {
        "official_allowed_usage": (
            "Publish a post, photo, or video to your Page. "
            "Update a post, photo, or video on your Page. "
            "Delete a post, photo, or video on your Page."
        ),
        "we_use": (
            "After an explicit Publish click, UploadM8 posts a recorded video to the "
            "Page the user administers via POST /{page-id}/videos (organic Page video / Reel)."
        ),
        "we_do_not": "Live streaming (publish_video), ads, or posting to Pages the user does not manage.",
        "app_review_notes": (
            "1) Functionality: Upload workflow — publish a user-selected recorded video to the "
            "Facebook Page the user administers.\n"
            "2) Integration: Page token + POST /{page-id}/videos. Dependencies pages_show_list and "
            "pages_read_engagement are requested with this permission. No post is created without "
            "the user clicking Publish.\n"
            "3) End-user value: Page admins publish organic Page video from UploadM8 and get the live link in Queue.\n"
            "Allowed usage match: Publish a video to your Page. We do not live-stream."
        ),
    },
    "pages_show_list": {
        "official_allowed_usage": (
            "Show a person the list of Pages they manage. Verify that a person manages a Page."
        ),
        "we_use": "GET /me/accounts at Facebook/Instagram connect (and token refresh) to pick the Page and store a Page token.",
        "we_do_not": "Listing or accessing Pages the user does not manage.",
        "app_review_notes": (
            "1) Functionality: Connected Accounts — Connect Facebook / Instagram.\n"
            "2) Integration: GET /me/accounts to show/verify the Page, then store that Page id + token. "
            "Requested as a dependency of pages_manage_posts and Instagram Professional connect.\n"
            "3) End-user value: The correct Page appears as the connected Facebook destination."
        ),
    },
    "pages_read_engagement": {
        "official_allowed_usage": (
            "Get content posted by your Page. Get names, PSIDs, and profile pictures of your Page followers. "
            "Get metadata about your Page."
        ),
        "we_use": "Read Page metadata (followers) and Page-owned video engagement fallbacks for the connected Page in Analytics.",
        "we_do_not": "Reading other users' private profiles or using follower PSIDs for ads.",
        "app_review_notes": (
            "1) Functionality: Analytics Page stats + dependency for pages_manage_posts / Instagram Graph.\n"
            "2) Integration: GET /{page-id}?fields=followers_count,fan_count and Page-owned video fields. "
            "Instagram connect also reads instagram_business_account on the Page.\n"
            "3) End-user value: Page admins see their own Page follower count and engagement in UploadM8."
        ),
    },
    "pages_read_user_content": {
        "official_allowed_usage": (
            "Get user generated content on your Page. Get posts that your Page is tagged in. "
            "Delete comments posted by users on your Page."
        ),
        "we_use": "GET /{page-id}/videos to list videos on the Page the user administers so Analytics can show their content.",
        "we_do_not": "Deleting comments, scraping other Pages, or redistributing user-generated content.",
        "app_review_notes": (
            "1) Functionality: Analytics → Platform Stats lists videos that exist on the user's Page.\n"
            "2) Integration: GET /{page-id}/videos (IDs/metadata) for the connected Page only, then insights on those IDs.\n"
            "3) End-user value: Creators review their own Page videos without leaving UploadM8.\n"
            "Allowed usage match: read content on the Page to help manage the Page. We do not delete comments."
        ),
    },
    "read_insights": {
        "official_allowed_usage": "Integrate Facebook's app, page or domain insights into your own analytics tools.",
        "we_use": "Show Page video insights (views, reactions, comments, shares) in the owner's Analytics dashboard.",
        "we_do_not": "Selling insights, sharing them with other customers, or ads optimization.",
        "app_review_notes": (
            "1) Functionality: Analytics for Facebook Page video performance.\n"
            "2) Integration: GET /{video-id}?fields=insights.metric(total_video_views,...) for videos on the connected Page.\n"
            "3) End-user value: Compare Facebook performance with other platforms in one dashboard.\n"
            "Allowed usage match: integrate Page insights into our analytics tool for the Page owner."
        ),
    },
    "instagram_basic": {
        "official_allowed_usage": (
            "Get basic metadata of an Instagram Business account profile, for example username and ID."
        ),
        "we_use": "At connect, read id/username/name/profile picture to label the connected Instagram account.",
        "we_do_not": "Reading other creators' private profiles.",
        "app_review_notes": (
            "1) Functionality: Connected Accounts identity for Instagram Professional.\n"
            "2) Integration: GET /{ig-user-id}?fields=id,username,name,profile_picture_url after resolving the IG account on the Page.\n"
            "3) End-user value: The correct Instagram username is shown in UploadM8."
        ),
    },
    "instagram_content_publish": {
        "official_allowed_usage": (
            "Managing organic content creation process for Instagram (for example, post photos and videos "
            "to main feed) on behalf of a business."
        ),
        "we_use": "Publish a user-selected video as an organic Instagram Reel after an explicit Publish click.",
        "we_do_not": "Ads, Stories-only APIs we do not support, or posting without user action.",
        "app_review_notes": (
            "1) Functionality: Upload — organic Instagram Reel publish.\n"
            "2) Integration: POST /{ig-user-id}/media then POST /{ig-user-id}/media_publish. No publish without Publish click.\n"
            "3) End-user value: Creators publish Reels from UploadM8 alongside other platforms."
        ),
    },
    "instagram_manage_insights": {
        "official_allowed_usage": (
            "Get metadata of an Instagram Business account. Get data insights of an Instagram Business account. "
            "Get story insights of an Instagram Business account."
        ),
        "we_use": "GET /{media-id}/insights for the connected account's Reels (plays, reach, saves, shares) in Analytics.",
        "we_do_not": "Insights for accounts the user did not connect.",
        "app_review_notes": (
            "1) Functionality: Analytics Instagram Reel performance.\n"
            "2) Integration: list /{ig-user-id}/media then /{media-id}/insights for that owner only.\n"
            "3) End-user value: Plays/reach/saves in one dashboard."
        ),
    },
    "business_management": {
        "official_allowed_usage": "Manage business assets such as an ad account. Claim ad accounts.",
        "we_use": (
            "Requested as a dependency so GET /me/accounts can list Pages available through Business Manager "
            "during Instagram connect. We do not manage ad accounts."
        ),
        "we_do_not": "Creating or claiming ad accounts, running ads, or Ads Manager APIs.",
        "app_review_notes": (
            "1) Functionality: Dependency of Instagram/Page connect (pages_show_list / Professional account linkage).\n"
            "2) Integration: Same GET /me/accounts used to list Pages the user can administer, including BM-linked Pages.\n"
            "3) End-user value: Agency/Business Manager Page admins can connect the correct Page.\n"
            "Main permission: pages_show_list / instagram_content_publish. We do not use ads APIs."
        ),
    },
}

META_GRAPH_RATE_LIMITING: Dict[str, Any] = {
    "docs": "https://developers.facebook.com/docs/graph-api/overview/rate-limiting",
    "token_class": (
        "Page access tokens use Business Use Case rate limits. "
        "User/app tokens use Platform rate limits (200 * DAU per rolling hour)."
    ),
    "headers_monitored": ["X-App-Usage", "X-Business-Use-Case-Usage"],
    "error_codes": [4, 17, 32, 613],
    "app_behavior": (
        "Outbound Meta Graph calls are gated (concurrent + spacing via outbound_slot meta). "
        "On Graph throttle (HTTP 429 or error 4/17/32/613) Facebook publish returns PLATFORM_RATE_LIMIT "
        "and does not retry in a tight loop. Analytics/catalog sync is batched and capped per worker cycle."
    ),
    "app_review_notes": (
        "UploadM8 respects Graph API rate limits. Page publish uses a Page access token (BUC limits). "
        "We throttle outbound Graph concurrency, space calls, and treat error codes 4, 17, 32, and 613 "
        "(and HTTP 429) as PLATFORM_RATE_LIMIT. We do not retry-storm when throttled. "
        "Analytics reads are batched (recent videos only) to stay within Platform/BUC budgets. "
        "Reference: https://developers.facebook.com/docs/graph-api/overview/rate-limiting"
    ),
}


_META_GRAPH_SLOT_DEPTH: contextvars.ContextVar[int] = contextvars.ContextVar(
    "meta_graph_slot_depth", default=0
)


@asynccontextmanager
async def meta_graph_slot():
    """Approved-usage throttle: all Graph calls go through the Meta outbound gate.

    Re-entrant in the same task so token refresh can call fetch_managed_pages.
    """
    depth = int(_META_GRAPH_SLOT_DEPTH.get() or 0)
    if depth > 0:
        yield
        return
    token = _META_GRAPH_SLOT_DEPTH.set(depth + 1)
    try:
        try:
            from stages.outbound_rl import outbound_slot
        except Exception:
            yield
            return
        async with outbound_slot("meta"):
            yield
    finally:
        _META_GRAPH_SLOT_DEPTH.reset(token)


# pages_manage_posts is Page VOD/Reels publish — not live streaming (publish_video).
FACEBOOK_GRAPH_PERMISSION_PROOF: List[Dict[str, str]] = [
    {
        "permission": "pages_manage_posts",
        "method": "POST",
        "path": "/{page-id}/videos",
        "purpose": "Publish a recorded Page video / Reel (organic VOD, not live)",
        "caller": "stages.publish_stage.publish_to_facebook",
    },
    {
        "permission": "pages_show_list",
        "method": "GET",
        "path": "/me/accounts",
        "purpose": "List Pages the user manages so we can store a Page access token",
        "caller": "services.meta_oauth.fetch_managed_pages",
    },
    {
        "permission": "pages_read_user_content",
        "method": "GET",
        "path": "/{page-id}/videos",
        "purpose": "List Page videos in Analytics and catalog sync",
        "caller": "routers.analytics._fetch_facebook_metrics; services.catalog_sync._list_facebook_videos",
    },
    {
        "permission": "read_insights",
        "method": "GET",
        "path": "/{video-id}?fields=insights.metric(total_video_views,...)",
        "purpose": "Page video view and engagement totals",
        "caller": "routers.analytics._fetch_facebook_metrics",
    },
    {
        "permission": "pages_read_engagement",
        "method": "GET",
        "path": "/{page-id}?fields=followers_count,fan_count",
        "purpose": "Page follower count plus engagement fallbacks on video objects",
        "caller": "routers.analytics._fetch_facebook_metrics",
    },
]

INSTAGRAM_GRAPH_PERMISSION_PROOF: List[Dict[str, str]] = [
    {
        "permission": "instagram_content_publish",
        "method": "POST",
        "path": "/{ig-user-id}/media and /{ig-user-id}/media_publish",
        "purpose": "Create and publish Instagram Reels",
        "caller": "stages.publish_stage.publish_to_instagram",
    },
    {
        "permission": "instagram_basic",
        "method": "GET",
        "path": "/{ig-user-id}?fields=id,username,name,profile_picture_url",
        "purpose": "Identify the Instagram Professional account at connect",
        "caller": "routers.oauth.oauth_callback",
    },
    {
        "permission": "instagram_manage_insights",
        "method": "GET",
        "path": "/{media-id}/insights",
        "purpose": "Reel plays, reach, saves, and shares in Analytics and per-upload sync",
        "caller": "routers.analytics._fetch_instagram_metrics; worker._sync_one_upload_analytics",
    },
    {
        "permission": "pages_show_list",
        "method": "GET",
        "path": "/me/accounts",
        "purpose": "Find the Facebook Page linked to the Instagram Business account",
        "caller": "services.meta_oauth.fetch_managed_pages",
    },
    {
        "permission": "pages_read_engagement",
        "method": "GET",
        "path": "/{page-id}?fields=instagram_business_account",
        "purpose": "Resolve the IG Professional account on the linked Page",
        "caller": "routers.oauth.oauth_callback",
    },
    {
        "permission": "pages_read_user_content",
        "method": "GET",
        "path": "/{page-id}?fields=instagram_business_account",
        "purpose": "instagram_basic dependency — read Page-linked IG Professional account during connect",
        "caller": "routers.oauth.oauth_callback",
    },
    {
        "permission": "business_management",
        "method": "GET",
        "path": "/me/accounts",
        "purpose": "List Pages available through Business Manager during Instagram connect",
        "caller": "services.meta_oauth.fetch_managed_pages",
    },
]

_META_GRAPH_RATE_LIMIT_CODES = frozenset({4, 17, 32, 613, 80004})

# Canonical Graph version for connect, publish, analytics, and catalog.
META_GRAPH_API_VERSION = "v21.0"


def split_oauth_scope(scope: str) -> List[str]:
    return [p.strip() for p in (scope or "").split(",") if p.strip()]


def facebook_pages_manage_posts_proof() -> Dict[str, str]:
    for row in FACEBOOK_GRAPH_PERMISSION_PROOF:
        if row.get("permission") == "pages_manage_posts":
            return dict(row)
    raise KeyError("pages_manage_posts proof row missing")


def facebook_page_videos_publish_url(page_id: str, *, version: str = META_GRAPH_API_VERSION) -> str:
    """Canonical Graph URL for pages_manage_posts Page video publish."""
    pid = str(page_id or "").strip()
    return f"https://graph.facebook.com/{version}/{pid}/videos"


def instagram_reels_container_url(ig_user_id: str, *, version: str = META_GRAPH_API_VERSION) -> str:
    """Canonical Graph URL for instagram_content_publish container create."""
    uid = str(ig_user_id or "").strip()
    return f"https://graph.facebook.com/{version}/{uid}/media"


def instagram_reels_publish_url(ig_user_id: str, *, version: str = META_GRAPH_API_VERSION) -> str:
    """Canonical Graph URL for instagram_content_publish media_publish."""
    uid = str(ig_user_id or "").strip()
    return f"https://graph.facebook.com/{version}/{uid}/media_publish"


def meta_graph_error_code(body: Any) -> Optional[int]:
    parsed: Any = body
    if isinstance(body, (bytes, bytearray)):
        try:
            body = body.decode("utf-8", errors="replace")
        except Exception:
            return None
    if isinstance(body, str):
        raw = body.strip()
        if not raw:
            return None
        try:
            parsed = json.loads(raw)
        except Exception:
            return None
    if not isinstance(parsed, dict):
        return None
    err = parsed.get("error")
    if not isinstance(err, dict):
        err = parsed
    code = err.get("code")
    try:
        return int(code) if code is not None else None
    except (TypeError, ValueError):
        return None


def meta_graph_is_rate_limited(
    status: int,
    body: Any = None,
    headers: Optional[Mapping[str, Any]] = None,
) -> bool:
    """True for Graph API throttle (codes 4/17/32/613) or HTTP 429."""
    try:
        code = int(status)
    except (TypeError, ValueError):
        code = 0
    if code == 429:
        return True
    if meta_graph_error_code(body) in _META_GRAPH_RATE_LIMIT_CODES:
        return True
    return _meta_usage_header_exhausted(headers)


def _meta_usage_header_exhausted(headers: Optional[Mapping[str, Any]]) -> bool:
    if not headers:
        return False
    folded = {str(k).lower(): v for k, v in headers.items()}
    for key in ("x-app-usage", "x-business-use-case-usage"):
        raw = folded.get(key)
        if raw is None or raw == "":
            continue
        parsed: Any = raw
        if isinstance(raw, (bytes, bytearray)):
            try:
                raw = raw.decode("utf-8", errors="replace")
            except Exception:
                continue
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except Exception:
                continue
        scores: List[float] = []

        def _collect(obj: Any) -> None:
            if not isinstance(obj, dict):
                return
            for metric in ("call_count", "total_cputime", "total_time"):
                try:
                    scores.append(float(obj.get(metric) or 0))
                except (TypeError, ValueError):
                    pass
            for v in obj.values():
                if isinstance(v, dict):
                    _collect(v)

        _collect(parsed)
        if any(s >= 100 for s in scores):
            return True
    return False


def meta_oauth_mode() -> str:
    m = (os.environ.get("META_OAUTH_MODE") or "full").strip().lower()
    if m in ("full", "minimal", "custom"):
        return m
    return "full"


def meta_instagram_oauth_scope() -> str:
    if meta_oauth_mode() == "minimal":
        return _SCOPES_MINIMAL
    if meta_oauth_mode() == "custom":
        return (os.environ.get("META_INSTAGRAM_OAUTH_SCOPE") or _SCOPES_INSTAGRAM_FULL).strip()
    return _SCOPES_INSTAGRAM_FULL


def meta_facebook_oauth_scope() -> str:
    if meta_oauth_mode() == "minimal":
        return _SCOPES_MINIMAL
    if meta_oauth_mode() == "custom":
        return (os.environ.get("META_FACEBOOK_OAUTH_SCOPE") or _SCOPES_FACEBOOK_FULL).strip()
    return _SCOPES_FACEBOOK_FULL


def meta_oauth_auth_type() -> str:
    """Facebook Login dialog flags.

    ``reauthenticate`` forces a fresh login (multi-account).
    ``rerequest`` re-prompts declined *and* newly added permissions — required after
    App Review so reconnect actually grants ``pages_manage_posts`` instead of
    silently reusing the old ``publish_video`` grant.
    """
    return "reauthenticate,rerequest"


def meta_app_credentials(platform: str | None = None) -> tuple[str, str]:
    """Client id/secret used to mint the token — must match the OAuth code exchange."""
    plat = str(platform or "").lower()
    if plat == "facebook":
        app_id = os.environ.get("FACEBOOK_CLIENT_ID") or os.environ.get("META_APP_ID") or ""
        app_secret = os.environ.get("FACEBOOK_CLIENT_SECRET") or os.environ.get("META_APP_SECRET") or ""
    else:
        app_id = os.environ.get("META_APP_ID") or os.environ.get("FACEBOOK_CLIENT_ID") or ""
        app_secret = os.environ.get("META_APP_SECRET") or os.environ.get("FACEBOOK_CLIENT_SECRET") or ""
    return app_id.strip(), app_secret.strip()


async def exchange_long_lived_user_token(
    client: httpx.AsyncClient,
    user_token: str,
    *,
    app_id: str,
    app_secret: str,
    graph_version: str = META_GRAPH_API_VERSION,
) -> tuple[str, Optional[int]]:
    """Exchange a short-lived user token for a ~60-day long-lived token.

    Page tokens from a short-lived user token expire in ~1–2 hours. Page tokens
    from a long-lived user token do not expire. On failure, returns the original
    token and ``None`` expires_in so callers can still proceed.
    """
    if not user_token or not app_id or not app_secret:
        return user_token, None
    try:
        resp = await client.get(
            f"https://graph.facebook.com/{graph_version}/oauth/access_token",
            params={
                "grant_type": "fb_exchange_token",
                "client_id": app_id,
                "client_secret": app_secret,
                "fb_exchange_token": user_token,
            },
            timeout=30.0,
        )
        if resp.status_code != 200:
            return user_token, None
        body = resp.json() or {}
        new_token = str(body.get("access_token") or "").strip()
        if not new_token:
            return user_token, None
        expires_in = body.get("expires_in")
        try:
            exp = int(float(expires_in)) if expires_in is not None else None
        except (TypeError, ValueError):
            exp = None
        return new_token, exp
    except Exception:
        return user_token, None


async def fetch_managed_pages(
    client: httpx.AsyncClient,
    user_access_token: str,
    *,
    fields: str = "id,name,username,access_token,picture",
    graph_version: str = META_GRAPH_API_VERSION,
    max_pages: int = 10,
) -> List[Dict[str, Any]]:
    """Paginated GET /me/accounts — first page alone misses Pages for some users."""
    if not user_access_token:
        return []
    out: List[Dict[str, Any]] = []
    url = f"https://graph.facebook.com/{graph_version}/me/accounts"
    params: Optional[Dict[str, Any]] = {
        "access_token": user_access_token,
        "fields": fields,
        "limit": 100,
    }
    try:
        async with meta_graph_slot():
            for _ in range(max(1, int(max_pages))):
                resp = await client.get(url, params=params, timeout=30.0)
                params = None
                if resp.status_code != 200:
                    break
                body = resp.json() or {}
                data = body.get("data") or []
                if isinstance(data, list):
                    out.extend(p for p in data if isinstance(p, dict))
                next_url = ((body.get("paging") or {}) if isinstance(body.get("paging"), dict) else {}).get("next")
                if not next_url:
                    break
                url = str(next_url)
    except Exception:
        return out
    return out


def pick_managed_page(
    pages: List[Dict[str, Any]],
    *,
    page_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    if not pages:
        return None
    want = str(page_id or "").strip()
    if want:
        for page in pages:
            if str(page.get("id") or "") == want:
                return page
        return None
    return pages[0]


def facebook_page_access_token_after_refresh(
    *,
    stored_access_token: str,
    stored_page_id: Optional[str],
    matched_page: Optional[Dict[str, Any]],
    new_user_token: str,
) -> str:
    """Page publish must use a Page token, never the user LLT, when we already have one."""
    matched_tok = str((matched_page or {}).get("access_token") or "").strip()
    if matched_tok:
        return matched_tok
    stored = str(stored_access_token or "").strip()
    if stored and str(stored_page_id or "").strip():
        return stored
    return str(new_user_token or "").strip() or stored


async def fetch_granted_permissions(
    client: httpx.AsyncClient,
    user_access_token: str,
    *,
    graph_version: str = META_GRAPH_API_VERSION,
) -> List[Dict[str, Any]]:
    """
    GET /me/permissions — returns [{"permission": "...", "status": "granted"|"declined"}, ...]
    """
    if not user_access_token:
        return []
    try:
        resp = await client.get(
            f"https://graph.facebook.com/{graph_version}/me/permissions",
            params={"access_token": user_access_token},
            timeout=20.0,
        )
        if resp.status_code != 200:
            return []
        data = resp.json() or {}
        raw = data.get("data") or []
        return raw if isinstance(raw, list) else []
    except Exception:
        return []


def permission_status(granted: List[Dict[str, Any]], permission: str) -> Optional[str]:
    """Return 'granted', 'declined', or None if not listed."""
    p = (permission or "").strip().lower()
    for row in granted:
        if str(row.get("permission") or "").lower() == p:
            return str(row.get("status") or "").lower() or None
    return None


def is_permission_granted(granted: List[Dict[str, Any]], permission: str) -> bool:
    return permission_status(granted, permission) == "granted"


def meta_permission_granted_from_blob(token_data: dict, permission: str) -> Optional[bool]:
    """
    True / False if we have a stored permission snapshot; None if unknown (legacy tokens).
    When None, callers may attempt the API call and surface Graph errors.
    """
    raw = token_data.get("meta_permissions")
    if not isinstance(raw, list) or not raw:
        return None
    st = permission_status(raw, permission)
    if st == "granted":
        return True
    if st == "declined":
        return False
    return None


def require_instagram_publish(token_data: dict) -> Optional[str]:
    """Return error message if publish cannot proceed; None if allowed or unknown."""
    g = meta_permission_granted_from_blob(token_data, "instagram_content_publish")
    if g is False:
        return (
            "Instagram publishing requires instagram_content_publish (not granted on this connection). "
            "Reconnect Instagram in Connected Accounts to grant the approved permission."
        )
    return None


def require_facebook_publish(token_data: dict) -> Optional[str]:
    """Page VOD/Reels require pages_manage_posts (not live-only publish_video)."""
    posts = meta_permission_granted_from_blob(token_data, "pages_manage_posts")
    if posts is True:
        return None
    if posts is False:
        return (
            "Facebook publishing requires pages_manage_posts (not granted on this connection). "
            "Reconnect Facebook in Connected Accounts to grant the approved Page publish permission. "
            "Note: publish_video is Meta live streaming only and does not authorize Page video/Reels."
        )
    return None
