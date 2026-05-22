"""
stages/product_card_renderer.py

Renders UploadM8 product card PNGs from catalog_products rows.

This module IS the canonical template generator. Everything else
(gen_cards_v2.py legacy, the sync script, Pikzels marketing wrapper)
calls render_card_bytes(row_dict) to get the base card.

The template format matches existing live cards (sub_agency.png style):
  800 x 500, M8 logo + UploadM8 wordmark, colored header band,
  black grid background, compliance footer with tax code + statement
  descriptor + cancellation/refund policy.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image, ImageDraw, ImageFont


# ---- Brand palette (sampled from existing live cards) ----
BG_DARK = (10, 9, 14)
CARD_BG = (18, 18, 26)
GRID = (28, 28, 38)
HEAD_BLUE = (59, 130, 246)
HEAD_ORANGE = (249, 115, 22)
HEAD_TEAL = (45, 212, 191)
WHITE = (255, 255, 255)
TEXT_DIM = (148, 163, 184)
TEXT_FAINT = (100, 116, 139)
TIER_TRIAL_BG = (34, 197, 94)


# ---- Geometry ----
W, H = 800, 500
CARD_X, CARD_Y = 30, 25
CARD_W = W - 2 * CARD_X
CARD_H = H - 2 * CARD_Y
CARD_RADIUS = 16
HEADER_H = 70


# ---- Font loader (cached) ----
_FONT_CACHE: Dict[tuple, ImageFont.FreeTypeFont] = {}


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    key = (size, bold)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    candidates_bold = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
    ]
    candidates_reg = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for path in (candidates_bold if bold else candidates_reg):
        try:
            font = ImageFont.truetype(path, size)
            _FONT_CACHE[key] = font
            return font
        except (OSError, IOError):
            continue
    fallback = ImageFont.load_default()
    _FONT_CACHE[key] = fallback
    return fallback


# ---- Cloud icon for header (loaded once, can be overridden by env var) ----
import os
_CLOUD_ICON_PATH = Path(os.environ.get(
    "UPLOADM8_CLOUD_ICON_PATH", "/var/www/uploadm8/images/cloud_icon.png"
))
_CLOUD_CACHE: Optional[Image.Image] = None


def _get_cloud() -> Optional[Image.Image]:
    global _CLOUD_CACHE
    if _CLOUD_CACHE is not None:
        return _CLOUD_CACHE
    if not _CLOUD_ICON_PATH.exists():
        return None
    img = Image.open(_CLOUD_ICON_PATH).convert("RGBA")
    ratio = 38 / img.height
    _CLOUD_CACHE = img.resize((int(img.width * ratio), 38), Image.LANCZOS)
    return _CLOUD_CACHE


# =============================================================
# Drawing primitives
# =============================================================

def _make_canvas() -> Image.Image:
    img = Image.new("RGB", (W, H), BG_DARK)
    d = ImageDraw.Draw(img)
    step = 28
    for x in range(0, W, step):
        d.line([(x, 0), (x, H)], fill=GRID, width=1)
    for y in range(0, H, step):
        d.line([(0, y), (W, y)], fill=GRID, width=1)
    return img


def _draw_card_body(img: Image.Image) -> ImageDraw.ImageDraw:
    d = ImageDraw.Draw(img)
    d.rounded_rectangle(
        (CARD_X, CARD_Y, CARD_X + CARD_W, CARD_Y + CARD_H),
        radius=CARD_RADIUS, fill=CARD_BG,
    )
    return d


def _draw_header_band(img: Image.Image, color: tuple) -> None:
    band = Image.new("RGBA", (CARD_W, HEADER_H), (0, 0, 0, 0))
    bd = ImageDraw.Draw(band)
    bd.rounded_rectangle(
        (0, 0, CARD_W, HEADER_H + CARD_RADIUS),
        radius=CARD_RADIUS, fill=color + (255,),
    )
    img.paste(band, (CARD_X, CARD_Y), band)


def _paste_logo(img: Image.Image) -> None:
    d = ImageDraw.Draw(img)
    f_m8 = _load_font(16, bold=True)
    f_word = _load_font(15, bold=True)
    pill_x = CARD_X + 22
    pill_y = CARD_Y + 22
    d.text((pill_x, pill_y + 4), "M8", font=f_m8, fill=WHITE)
    d.text((pill_x + 38, pill_y + 5), "UploadM8", font=f_word, fill=WHITE)
    cloud = _get_cloud()
    if cloud:
        img.paste(cloud, (CARD_X + CARD_W - cloud.width - 22, CARD_Y + 16), cloud)


def _draw_trial_pill(img: Image.Image) -> None:
    d = ImageDraw.Draw(img)
    f = _load_font(13, bold=True)
    text = "7-DAY FREE TRIAL"
    bbox = d.textbbox((0, 0), text, font=f)
    tw = bbox[2] - bbox[0]
    pad = 14
    pw, ph = tw + pad * 2, 28
    px = CARD_X + CARD_W - pw - 22
    py = CARD_Y + 22
    d.rounded_rectangle((px, py, px + pw, py + ph), radius=14, fill=TIER_TRIAL_BG)
    d.text((px + pad, py + 7), text, font=f, fill=WHITE)


def _draw_footer(img: Image.Image, kind: str, statement: str, tax_code: str) -> None:
    d = ImageDraw.Draw(img)
    f_tiny = _load_font(13)
    footer_y = CARD_Y + CARD_H - 70
    d.line([(CARD_X + 24, footer_y - 12),
            (CARD_X + CARD_W - 24, footer_y - 12)],
           fill=(60, 60, 75), width=1)
    if kind == "subscription":
        l1 = f"DIGITAL SERVICE   .   SAAS SUBSCRIPTION   .   TAX CODE: {tax_code}"
        l2 = "uploadm8.com   .   support@uploadm8.com   .   Multi-platform video publishing service"
        l3 = f"Statement descriptor: {statement}   .   Billed monthly via Stripe"
    else:
        l1 = f"DIGITAL TOKEN  .  ONE-TIME PURCHASE  .  TAX CODE: {tax_code}  .  Non-refundable"
        l2 = "Tokens added to your UploadM8 wallet immediately upon payment. Never expire."
        l3 = f"uploadm8.com   .   support@uploadm8.com   .   Statement: {statement}"
    d.text((CARD_X + 22, footer_y - 4), l1, font=f_tiny, fill=TEXT_FAINT)
    d.text((CARD_X + 22, footer_y + 14), l2, font=f_tiny, fill=TEXT_FAINT)
    d.text((CARD_X + 22, footer_y + 32), l3, font=f_tiny, fill=TEXT_FAINT)


# =============================================================
# Card variants
# =============================================================

def _render_tier_card(row: Dict[str, Any]) -> Image.Image:
    img = _make_canvas()
    _draw_card_body(img)

    # Creator Pro uses orange header in your live template; rest use blue.
    is_pro = (row.get("tier_slug") == "creator_pro"
              or row.get("display_name", "").lower() == "creator pro")
    header_color = HEAD_ORANGE if is_pro else HEAD_BLUE
    _draw_header_band(img, header_color)
    _paste_logo(img)

    price = float(row.get("price_usd") or 0)
    if price > 0:
        _draw_trial_pill(img)

    d = ImageDraw.Draw(img)
    f_title = _load_font(48, bold=True)
    f_price = _load_font(40, bold=True)
    f_med = _load_font(18)
    f_med_bold = _load_font(20, bold=True)
    f_small = _load_font(14)

    y = CARD_Y + HEADER_H + 22
    d.text((CARD_X + 22, y), row["display_name"], font=f_title, fill=WHITE)

    accent = HEAD_ORANGE if is_pro else HEAD_BLUE
    price_text = f"${price:.2f}" if price > 0 else "FREE"
    d.text((CARD_X + 22, y + 58), price_text, font=f_price, fill=accent)
    sub = ("per month  .  recurring  .  cancel anytime" if price > 0 else "forever")
    d.text((CARD_X + 22, y + 108), sub, font=f_med, fill=TEXT_DIM)

    div_y = y + 145
    d.line([(CARD_X + 22, div_y), (CARD_X + CARD_W - 22, div_y)],
           fill=(60, 60, 75), width=1)

    stats_y = div_y + 22
    cell_w = (CARD_W - 44) // 4
    qd = int(row.get("queue_depth") or 0)
    queue_text = "infinite" if qd >= 99999 else f"{qd:,}"
    cells = [
        (str(int(row.get("max_accounts") or 0)), "ACCOUNTS/PLATFORM"),
        (f"{int(row.get('put_monthly') or 0):,}", "PUT / MONTH"),
        (f"{int(row.get('aic_monthly') or 0):,}", "AIC / MONTH"),
        (queue_text, "QUEUE DEPTH"),
    ]
    for i, (v, l) in enumerate(cells):
        cx = CARD_X + 22 + i * cell_w
        d.text((cx, stats_y), v, font=f_price, fill=WHITE)
        d.text((cx, stats_y + 48), l, font=f_small, fill=TEXT_DIM)

    strip_y = stats_y + 80
    look = int(row.get("lookahead_hours") or 0)
    d.text(
        (CARD_X + 22, strip_y),
        f"Scheduling lookahead: {look}h   .   "
        f"TikTok  .  YouTube Shorts  .  Instagram Reels  .  Facebook Reels",
        font=f_small, fill=TEXT_DIM,
    )

    statement = row.get("statement_descriptor") or f"UPLOADM8 {row['display_name'].upper()}"
    _draw_footer(img, "subscription", statement, row.get("tax_code") or "txcd_10103001")
    return img


def _render_topup_card(row: Dict[str, Any]) -> Image.Image:
    img = _make_canvas()
    _draw_card_body(img)
    wallet = (row.get("wallet") or "").upper()
    accent = HEAD_ORANGE if wallet == "PUT" else HEAD_TEAL
    _draw_header_band(img, accent)
    _paste_logo(img)

    d = ImageDraw.Draw(img)
    f_huge = _load_font(54, bold=True)
    f_price = _load_font(40, bold=True)
    f_med_bold = _load_font(20, bold=True)
    f_med = _load_font(18)
    f_small = _load_font(14)

    amount = int(row.get("token_amount") or 0)
    price = float(row.get("price_usd") or 0)
    cx = CARD_X + CARD_W // 2
    y0 = CARD_Y + HEADER_H + 35

    s = f"{amount:,}"
    bbox = d.textbbox((0, 0), s, font=f_huge)
    d.text((cx - (bbox[2] - bbox[0]) // 2, y0), s, font=f_huge, fill=WHITE)

    label = f"{wallet} TOKENS"
    bbox = d.textbbox((0, 0), label, font=f_med_bold)
    d.text((cx - (bbox[2] - bbox[0]) // 2, y0 + 70), label, font=f_med_bold, fill=accent)

    sub = "Upload Credits" if wallet == "PUT" else "AI Credits"
    bbox = d.textbbox((0, 0), sub, font=f_med)
    d.text((cx - (bbox[2] - bbox[0]) // 2, y0 + 100), sub, font=f_med, fill=TEXT_DIM)

    price_text = f"${price:.2f}"
    bbox = d.textbbox((0, 0), price_text, font=f_price)
    d.text((cx - (bbox[2] - bbox[0]) // 2, y0 + 138), price_text, font=f_price, fill=accent)

    if wallet == "PUT":
        vp = f"Publish {amount:,} videos across all connected platforms"
    else:
        vp = f"AI-powered optimization for {amount:,} uploads"
    bbox = d.textbbox((0, 0), vp, font=f_small)
    d.text((cx - (bbox[2] - bbox[0]) // 2, y0 + 190), vp, font=f_small, fill=TEXT_DIM)

    statement = row.get("statement_descriptor") or f"UPLOADM8 {wallet} {amount}"
    _draw_footer(img, "topup", statement, row.get("tax_code") or "txcd_10103001")
    return img


# =============================================================
# Public API
# =============================================================

def render_card_image(row: Dict[str, Any]) -> Image.Image:
    """Render a PIL Image for this catalog row."""
    kind = row.get("product_kind", "subscription")
    if kind == "subscription":
        return _render_tier_card(row)
    if kind in ("topup_put", "topup_aic"):
        return _render_topup_card(row)
    raise ValueError(f"Unknown product_kind: {kind!r}")


def render_card_bytes(row: Dict[str, Any]) -> bytes:
    """Render and return PNG bytes."""
    img = render_card_image(row)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def render_card_to_path(row: Dict[str, Any], output_path: Path) -> None:
    img = render_card_image(row)
    img.save(output_path, format="PNG", optimize=True)
