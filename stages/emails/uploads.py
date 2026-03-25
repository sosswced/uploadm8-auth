"""
UploadM8 — Phase 4a: Upload Notification Emails  (v2 — Enhanced Design)
=========================================================================
  send_upload_completed_email  → upload worker: job reaches succeeded/completed status
  send_upload_failed_email     → upload worker: job reaches failed status

Both respect user_preferences.email_notifications.

v2 upgrades:
  - Both emails have preheader_text
  - Completed: section_tag "Upload Live", metric_hero for platform count, improved badges
  - Failed: section_tag "Upload Failed", enhanced error display
"""

import logging
import html as _html
from .base import (
    send_email, mailgun_ready,
    email_shell, intro_row, body_row, cta_button, tinted_box,
    check_list, stat_grid, secondary_links, alert_banner, spacer,
    section_tag, metric_hero, divider_accent,
    GRAD_GREEN, GRAD_RED, GRAD_ORANGE,
    URL_DASHBOARD, SUPPORT_EMAIL,
)

logger = logging.getLogger("uploadm8-worker")

# Platform brand colours (used in mini-badges)
PLATFORM_COLORS = {
    "tiktok":    "#1a1a1a",
    "youtube":   "#ff0000",
    "instagram": "#c13584",
    "facebook":  "#1877f2",
}

PLATFORM_NAMES = {
    "tiktok":    "TikTok",
    "youtube":   "YouTube Shorts",
    "instagram": "Instagram Reels",
    "facebook":  "Facebook Reels",
}

def _escape_html(value: object) -> str:
    return _html.escape(str(value or ""), quote=False).replace("\n", "<br/>")

def _get_result_field(result: object, field: str, default: object = None) -> object:
    """
    Allows helpers to work with both dataclass-like PlatformResult objects and
    dict-like fallback shapes.
    """
    if isinstance(result, dict):
        return result.get(field, default)
    return getattr(result, field, default)


def _format_hashtags(raw_tags: object) -> str:
    """
    Normalizes hashtags into a single string like:
      "#tag1 #tag2"
    """
    if not raw_tags:
        return ""
    if isinstance(raw_tags, str):
        # Accept "tag1 tag2" or "#tag1,#tag2"
        raw_tags = [t.strip() for t in raw_tags.replace(",", " ").split() if t.strip()]

    parts: list[str] = []
    for t in raw_tags if isinstance(raw_tags, list) else [raw_tags]:
        s = str(t).strip()
        if not s:
            continue
        s = s if s.startswith("#") else f"#{s}"
        parts.append(s)
    return " ".join(parts)


def _platform_view_url(result: object) -> str | None:
    """
    Best-effort external post URL builder, matching the Discord notification logic.
    """
    if not bool(_get_result_field(result, "success", False)):
        return None

    url = _get_result_field(result, "platform_url", None)
    if url and str(url).startswith("http"):
        return str(url)

    plat = (str(_get_result_field(result, "platform", "") or "") or "").lower()
    vid = _get_result_field(result, "platform_video_id", None)
    handle = str(_get_result_field(result, "account_username", None) or "")

    if not vid:
        return None

    if plat == "tiktok" and handle:
        h = str(handle).lstrip("@")
        return f"https://www.tiktok.com/@{h}/video/{vid}"
    if plat == "youtube":
        return f"https://www.youtube.com/shorts/{vid}"
    if plat == "facebook":
        return f"https://www.facebook.com/video/{vid}"

    # Instagram & others: platform_url should already be present.
    return None


def _platform_chips_with_links(platform_results: list[object]) -> str:
    """
    Renders platform chips similar to dashboard, but inside a single email row.
    Chips link to the external post URL when available; otherwise they render dashed.
    """
    chips = []
    for r in platform_results or []:
        p = (str(_get_result_field(r, "platform", "") or "")).lower() or "unknown"
        color = PLATFORM_COLORS.get(p, "#374151")
        label_base = PLATFORM_NAMES.get(p, p.title())

        username = _get_result_field(r, "account_username", None) or None
        account_name = _get_result_field(r, "account_name", None) or None
        avatar_url = _get_result_field(r, "account_avatar", None) or None

        if username:
            u = str(username).lstrip("@").strip() or None
            disp_label = f"@{u}" if u else label_base
        elif account_name:
            disp_label = str(account_name).strip()
        else:
            disp_label = label_base

        avatar_html = (
            f'<img src="{_escape_html(avatar_url)}" alt="" '
            f'style="width:14px;height:14px;border-radius:50%;object-fit:cover;'
            f'margin-right:6px;vertical-align:middle;" />'
            if avatar_url
            else ""
        )

        url = _platform_view_url(r)
        is_link = bool(url)
        success = bool(_get_result_field(r, "success", False))

        # Dashboard-like treatment: only link chips for successful platforms.
        if is_link:
            chip = (
                f'<a href="{url}" target="_blank" rel="noopener" '
                f'style="display:inline-block;background:{color};color:#ffffff;'
                f'font-size:12px;font-weight:700;padding:5px 14px;border-radius:99px;'
                f'margin:4px 5px;border:1px solid rgba(255,255,255,0.12);'
                f'text-decoration:none;">'
                f'{avatar_html}{_escape_html(disp_label)} &#8599;'
                f'</a>'
            )
        else:
            # No link yet (or failed): dim and render dashed border.
            chip = (
                f'<span style="display:inline-block;background:rgba(255,255,255,0.06);'
                f'color:#ffffff;font-size:12px;font-weight:700;padding:5px 14px;'
                f'border-radius:99px;margin:4px 5px;'
                f'border:1px dashed rgba(255,255,255,0.18);opacity:{0.65 if success else 0.45};'
                f'cursor:default;">'
                f'{avatar_html}{_escape_html(disp_label)}'
                f'</span>'
            )

        chips.append(chip)

    chips_html = "".join(chips)
    return f'<tr><td style="padding:0 40px 28px;text-align:center;">{chips_html}</td></tr>'


def _post_details_block(video_title: str, video_caption: str, video_hashtags: object) -> str:
    tags_str = _format_hashtags(video_hashtags)
    has_caption = bool(str(video_caption or "").strip())

    caption_header = (
        "<div style='margin:0 0 14px;color:#9ca3af;'>📝 Caption</div>"
        if has_caption else ""
    )
    caption_body = (
        f"<div style='margin:0 0 14px;color:#e5e7eb;font-size:14px;'>"
        f"{_escape_html(video_caption)}</div>"
        if has_caption else
        "<div style='margin:0 0 14px;color:#9ca3af;font-style:italic;'>No caption provided</div>"
    )

    hashtags_header = (
        "<div style='margin:0 0 10px;color:#9ca3af;font-size:12px;letter-spacing:.12em;"
        "font-weight:800;text-transform:uppercase;'>🏷️ Hashtags</div>"
        if tags_str else ""
    )
    hashtags_body = (
        f"<div style='margin:0;color:#e5e7eb;font-size:13px;line-height:1.65;'>"
        f"{_escape_html(tags_str)}</div>"
        if tags_str else
        "<div style='margin:0;color:#9ca3af;font-style:italic;'>No hashtags</div>"
    )

    inner_html = (
        "<div style='color:#ffffff;font-size:14px;line-height:1.65;'>"
        "<div style='margin:0 0 10px;color:#9ca3af;font-size:12px;letter-spacing:.12em;font-weight:800;text-transform:uppercase;'>"
        "📹 Title</div>"
        f"<div style='margin:0 0 14px;font-size:18px;font-weight:900;'>{_escape_html(video_title)}</div>"
        f"{caption_header}{caption_body}"
        f"{hashtags_header}{hashtags_body}"
        "</div>"
    )

    return tinted_box(inner_html=inner_html, hex_color="#f97316")


def _platform_results_list(platform_results: list[object]) -> str:
    """
    Renders a per-platform "View Post" / error line list (Discord-like fields).
    """
    blocks = []
    for r in platform_results or []:
        p = (str(_get_result_field(r, "platform", "") or "")).lower() or "unknown"
        plat_name = PLATFORM_NAMES.get(p, p.title())

        icon = "✅" if bool(_get_result_field(r, "success", False)) else "❌"
        username = _get_result_field(r, "account_username", None) or ""
        if username:
            u = str(username).lstrip("@").strip()
            account_label = f" @{_escape_html(u)}"
        else:
            account_label = ""

        url = _platform_view_url(r)
        if url and bool(_get_result_field(r, "success", False)):
            value_html = (
                f'<a href="{url}" target="_blank" rel="noopener" '
                f'style="color:#f97316;text-decoration:none;font-weight:900;">View Post &#8599;</a>'
            )
        else:
            raw_err = (
                _get_result_field(r, "error_message", None)
                or _get_result_field(r, "error_code", None)
                or "Unknown error"
            )
            value_html = _escape_html(raw_err)[:160]

        blocks.append(
            f"<div style='margin:0 0 8px;color:#e5e7eb;font-size:13px;line-height:1.6;'>"
            f"<span style='color:#ffffff;font-weight:900;'>{icon} {plat_name}</span>"
            f"{account_label}: {value_html}"
            f"</div>"
        )

    inner = "".join(blocks) if blocks else "<div style='color:#9ca3af;font-style:italic;'>No platform results</div>"
    return body_row(inner, padding="0 40px 28px")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Upload completed
# ─────────────────────────────────────────────────────────────────────────────
async def send_upload_completed_email(
    email: str,
    name: str,
    filename: str,
    platforms: list[str],
    put_spent: int = 0,
    aic_spent: int = 0,
    upload_id: str = "",
    duration_seconds: int = 0,
    video_title: str | None = None,
    video_caption: str | None = None,
    video_hashtags: object = None,
    platform_results: list[object] | None = None,
) -> None:
    """
    Sent when an upload job reaches 'succeeded' or 'completed' status.
    Only fires if user_preferences.email_notifications is True (checked by caller).

    Usage in worker finish handler:
        prefs = await conn.fetchrow(
            "SELECT email_notifications FROM user_preferences WHERE user_id=$1", user_id
        )
        if prefs and prefs["email_notifications"]:
            await send_upload_completed_email(email, name, filename, platforms, ...)
    """
    if not mailgun_ready():
        return

    platform_results = platform_results or []
    success_results = [r for r in platform_results if bool(_get_result_field(r, "success", False))]
    failed_results  = [r for r in platform_results if not bool(_get_result_field(r, "success", False))]

    platform_count = len(success_results) if platform_results else len(platforms)
    platform_word  = "platform" if platform_count == 1 else "platforms"
    is_partial = bool(failed_results) and bool(success_results)
    dur_label = f"{duration_seconds}s" if duration_seconds else "—"

    token_stats = []
    if put_spent:
        token_stats.append(("PUT Spent", str(put_spent)))
    if aic_spent:
        token_stats.append(("AIC Spent", str(aic_spent)))
    token_stats.append(("Platforms", str(platform_count)))

    _title = video_title or filename
    _caption = video_caption or ""
    _hashtags = video_hashtags or []

    _gradient = GRAD_ORANGE if is_partial else GRAD_GREEN
    _section_color = "#f97316" if is_partial else "#16a34a"
    _section_label = "Partial Upload" if is_partial else "Upload Live &#127775;"

    preheader_suffix = "partially live" if is_partial else "live"

    html = email_shell(
        gradient=_gradient,
        tagline="Upload once. Publish everywhere.",
        preheader_text=(
            f"{filename} is {preheader_suffix}! "
            f"Published to {platform_count} {platform_word} and reaching your audience now."
        ),
        body_rows=(
            section_tag(_section_label, _section_color)
            + intro_row(
                "Your upload is live!"
                + (" &#9888;&#65039;" if is_partial else " &#127775;"),
                f"<strong style='color:#ffffff;'>{_escape_html(filename)}</strong> has been "
                f"published to <strong style='color:#22c55e;'>{platform_count} {platform_word}</strong>. "
                + ("Some platforms failed — you can retry from your dashboard." if is_partial else "Your content is now reaching your audience."),
            )
            + metric_hero(
                str(platform_count),
                f"Platform{'s' if platform_count != 1 else ''} "
                + ("Live (Partial)" if is_partial else "Live"),
                f"{filename}",
                _section_color,
            )
            + _post_details_block(_title, _caption, _hashtags)
            + _platform_chips_with_links(platform_results)
            + _platform_results_list(platform_results)
            + stat_grid(*token_stats)
            + cta_button("View Upload Details", URL_DASHBOARD, pt="20px", pb="20px")
            + secondary_links(("Dashboard", URL_DASHBOARD),)
        ),
        footer_note="You received this because upload email notifications are enabled on your account.",
    )

    await send_email(
        email,
        f"🌟 Your upload is live on {platform_count} {platform_word}!",
        html,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Upload failed
# ─────────────────────────────────────────────────────────────────────────────
async def send_upload_failed_email(
    email: str,
    name: str,
    filename: str,
    platforms: list[str],
    error_reason: str = "",
    upload_id: str = "",
    stage: str = "",
    video_title: str | None = None,
    video_caption: str | None = None,
    video_hashtags: object = None,
    platform_results: list[object] | None = None,
) -> None:
    """
    Sent when an upload job reaches 'failed' status.
    Only fires if user_preferences.email_notifications is True (checked by caller).

    stage examples: "ingest", "transcode", "publish", "platform_api"
    """
    if not mailgun_ready():
        return

    reason_display = error_reason or "An unexpected error occurred during processing."
    stage_label    = stage.replace("_", " ").title() if stage else "Processing"

    platform_results = platform_results or []
    _title = video_title or filename
    _caption = video_caption or ""
    _hashtags = video_hashtags or []

    # If we don't have per-platform results, fall back to platform names.
    if not platform_results and platforms:
        platform_results = [{"platform": p, "success": False} for p in platforms]  # type: ignore[list-item]

    html = email_shell(
        gradient=GRAD_RED,
        tagline="Upload once. Publish everywhere.",
        preheader_text=f"Upload failed: {filename} could not be published. Your tokens have been refunded.",
        body_rows=(
            section_tag("Upload Failed", "#ef4444")
            + intro_row(
                "Upload failed &#10060;",
                f"Unfortunately, <strong style='color:#ffffff;'>{filename}</strong> "
                f"could not be published. The error occurred during the "
                f"<strong style='color:#f87171;'>{stage_label}</strong> stage.",
            )
            + _post_details_block(_title, _caption, _hashtags)
            + _platform_chips_with_links(platform_results)
            + _platform_results_list(platform_results)
            + tinted_box(
                f'<p style="margin:0 0 6px;color:#6b7280;font-size:10px;text-transform:uppercase;'
                f'letter-spacing:1.2px;font-weight:600;">Error Details</p>'
                f'<p style="margin:0;color:#f87171;font-size:14px;line-height:1.65;'
                f'font-family:\'Courier New\',Courier,monospace;">{reason_display}</p>'
                + (
                    f'<p style="margin:10px 0 0;color:#6b7280;font-size:12px;">'
                    f'Upload ID: <code style="color:#f97316;">{upload_id}</code></p>'
                    if upload_id else ""
                ),
                hex_color="#ef4444",
            )
            + check_list(
                "Tokens used for this upload have been refunded",
                "Your other uploads are not affected",
                "You can retry the upload from your dashboard",
                hex_color="#22c55e",
            )
            + cta_button("Retry Upload", URL_DASHBOARD, pt="4px", pb="20px")
            + tinted_box(
                f'<p style="margin:0;color:#9ca3af;font-size:13px;line-height:1.65;">'
                f'If this keeps happening, contact us at '
                f'<a href="mailto:{SUPPORT_EMAIL}" style="color:#f97316;text-decoration:none;">'
                f'{SUPPORT_EMAIL}</a> with your upload ID: '
                f'<code style="color:#f97316;font-size:12px;">{upload_id or "—"}</code></p>',
                hex_color="#374151",
                pb="36px",
            )
        ),
        footer_note="You received this because upload email notifications are enabled on your account.",
    )

    await send_email(email, f"❌ Upload failed — {filename}", html)
