"""
UploadM8 Stripe product card PNGs (800×500). Shared by ``scripts/generate_product_cards.py``
and the billing catalog sync job.

Canonical filenames under ``out_dir``:
  Tier:  sub_starter.png, sub_creator_lite.png, sub_creator_pro.png,
         sub_studio.png, sub_agency.png
  PUT:   topup_put_50.png … topup_put_1000.png
  AIC:   topup_aic_50.png … topup_aic_1000.png
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

# ---- Brand palette (sampled from existing template assets) ----
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

# Card-only top-up prices when no ``uploadm8_{put|aic}_{n}`` row exists yet
_CARD_TOPUP_PRICE_FALLBACK: dict[tuple[str, int], float] = {
    ("put", 50): 2.99,
    ("put", 100): 4.99,
    ("aic", 50): 2.99,
    ("aic", 100): 4.99,
}

W, H = 800, 500
CARD_X, CARD_Y = 30, 25
CARD_W = W - 2 * CARD_X
CARD_H = H - 2 * CARD_Y
CARD_RADIUS = 16
HEADER_H = 70

_PUBLIC_CARD_SLUGS = ("free", "creator_lite", "creator_pro", "studio", "agency")
_TOPUP_AMOUNTS = (50, 100, 250, 500, 1000)


def _font_paths(bold: bool) -> list[Path]:
    win = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    return [
        win / ("segoeuib.ttf" if bold else "segoeui.ttf"),
        win / ("arialbd.ttf" if bold else "arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
        if bold
        else Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf")
        if bold
        else Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
        Path("/Library/Fonts/Arial Bold.ttf") if bold else Path("/Library/Fonts/Arial.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")
        if bold
        else Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    ]


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    for p in _font_paths(bold):
        try:
            return ImageFont.truetype(str(p), size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


F_TITLE = load_font(48, bold=True)
F_PRICE = load_font(40, bold=True)
F_HUGE = load_font(54, bold=True)
F_MED_BOLD = load_font(20, bold=True)
F_MED = load_font(18)
F_SMALL = load_font(14)
F_TINY_BOLD = load_font(13, bold=True)
F_TINY = load_font(13)
F_LOGO_M8 = load_font(16, bold=True)
F_LOGO_TEXT = load_font(15, bold=True)


def make_canvas() -> Image.Image:
    img = Image.new("RGB", (W, H), BG_DARK)
    draw = ImageDraw.Draw(img)
    step = 28
    for x in range(0, W, step):
        draw.line([(x, 0), (x, H)], fill=GRID, width=1)
    for y in range(0, H, step):
        draw.line([(0, y), (W, y)], fill=GRID, width=1)
    return img


def draw_card_body(img: Image.Image) -> None:
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle(
        (CARD_X, CARD_Y, CARD_X + CARD_W, CARD_Y + CARD_H),
        radius=CARD_RADIUS,
        fill=CARD_BG,
    )


def draw_header_band(img: Image.Image, color: tuple[int, int, int]) -> None:
    band = Image.new("RGBA", (CARD_W, HEADER_H), (0, 0, 0, 0))
    bd = ImageDraw.Draw(band)
    bd.rounded_rectangle(
        (0, 0, CARD_W, HEADER_H + CARD_RADIUS),
        radius=CARD_RADIUS,
        fill=color + (255,),
    )
    img.paste(band, (CARD_X, CARD_Y), band)


def _draw_cloud_glyph(img: Image.Image, cx: int, cy: int, scale: float = 1.0) -> None:
    """Simple white cloud (three circles) when no PNG asset is available."""
    draw = ImageDraw.Draw(img, "RGBA")
    r = int(9 * scale)
    offsets = ((-14, 4), (0, 0), (14, 4))
    for dx, dy in offsets:
        x0, y0 = cx + dx - r, cy + dy - r
        x1, y1 = cx + dx + r, cy + dy + r
        draw.ellipse((x0, y0, x1, y1), fill=WHITE + (255,))


def paste_logo(img: Image.Image, cloud_icon_path: Path | None) -> None:
    draw = ImageDraw.Draw(img)
    pill_x = CARD_X + 22
    pill_y = CARD_Y + 22
    draw.text((pill_x, pill_y + 4), "M8", font=F_LOGO_M8, fill=WHITE)
    draw.text((pill_x + 38, pill_y + 5), "UploadM8", font=F_LOGO_TEXT, fill=WHITE)

    ix = CARD_X + CARD_W - 22
    iy = CARD_Y + 16 + 19
    if cloud_icon_path and cloud_icon_path.is_file():
        try:
            cloud = Image.open(cloud_icon_path).convert("RGBA")
            ratio = 38 / max(cloud.height, 1)
            nw = max(1, int(cloud.width * ratio))
            cloud = cloud.resize((nw, 38), Image.Resampling.LANCZOS)
            img.paste(cloud, (CARD_X + CARD_W - cloud.width - 22, CARD_Y + 16), cloud)
            return
        except OSError:
            pass
    _draw_cloud_glyph(img, ix - 22, iy - 10, scale=1.15)


def draw_trial_pill(img: Image.Image, text: str = "7-DAY FREE TRIAL") -> None:
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), text, font=F_TINY_BOLD)
    tw = bbox[2] - bbox[0]
    pad = 14
    pill_w = tw + pad * 2
    pill_h = 28
    px = CARD_X + CARD_W - pill_w - 22
    py = CARD_Y + 22
    draw.rounded_rectangle(
        (px, py, px + pill_w, py + pill_h),
        radius=14,
        fill=TIER_TRIAL_BG,
    )
    draw.text((px + pad, py + 7), text, font=F_TINY_BOLD, fill=WHITE)


def draw_footer(
    img: Image.Image,
    tax_code: str,
    statement: str,
    kind: str,
) -> None:
    draw = ImageDraw.Draw(img)
    footer_y = CARD_Y + CARD_H - 70
    draw.line(
        [(CARD_X + 24, footer_y - 12), (CARD_X + CARD_W - 24, footer_y - 12)],
        fill=(60, 60, 75),
        width=1,
    )

    if kind == "subscription":
        line1 = f"DIGITAL SERVICE   ·   SAAS SUBSCRIPTION   ·   TAX CODE: {tax_code}"
        line2 = "uploadm8.com   ·   support@uploadm8.com   ·   Multi-platform video publishing service"
        line3 = f"Statement descriptor: {statement}   ·   Billed monthly via Stripe"
    else:
        line1 = (
            f"DIGITAL TOKEN  ·  ONE-TIME PURCHASE  ·  TAX CODE: {tax_code}  ·  Non-refundable"
        )
        line2 = "Tokens added to your UploadM8 wallet immediately upon payment. Never expire."
        line3 = f"uploadm8.com   ·   support@uploadm8.com   ·   Statement: {statement}"

    draw.text((CARD_X + 22, footer_y - 4), line1, font=F_TINY, fill=TEXT_FAINT)
    draw.text((CARD_X + 22, footer_y + 14), line2, font=F_TINY, fill=TEXT_FAINT)
    draw.text((CARD_X + 22, footer_y + 32), line3, font=F_TINY, fill=TEXT_FAINT)


def draw_stat_cell(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    value: str,
    label: str,
) -> None:
    draw.text((x, y), value, font=F_PRICE, fill=WHITE)
    draw.text((x, y + 48), label, font=F_SMALL, fill=TEXT_DIM)


def make_tier_card(
    name: str,
    price: float,
    put_monthly: int,
    aic_monthly: int,
    max_accounts: int,
    queue_depth: int,
    lookahead_h: int,
    cloud_path: Path | None,
    *,
    accent_orange: bool,
    show_trial_pill: bool,
    tax_code: str = "txcd_10103001",
) -> Image.Image:
    img = make_canvas()
    draw_card_body(img)
    draw_header_band(img, HEAD_ORANGE if accent_orange else HEAD_BLUE)
    paste_logo(img, cloud_path)
    if show_trial_pill:
        draw_trial_pill(img)
    draw = ImageDraw.Draw(img)

    y = CARD_Y + HEADER_H + 22
    draw.text((CARD_X + 22, y), name, font=F_TITLE, fill=WHITE)

    accent = HEAD_ORANGE if accent_orange else HEAD_BLUE
    price_text = f"${price:.2f}" if price > 0 else "FREE"
    draw.text((CARD_X + 22, y + 58), price_text, font=F_PRICE, fill=accent)
    sub = "per month  ·  recurring  ·  cancel anytime" if price > 0 else "forever"
    draw.text((CARD_X + 22, y + 108), sub, font=F_MED, fill=TEXT_DIM)

    div_y = y + 145
    draw.line(
        [(CARD_X + 22, div_y), (CARD_X + CARD_W - 22, div_y)],
        fill=(60, 60, 75),
        width=1,
    )

    stats_y = div_y + 22
    cell_w = (CARD_W - 44) // 4
    queue_text = "∞" if queue_depth >= 9999 else f"{queue_depth:,}"
    cells = [
        (str(max_accounts), "ACCOUNTS/PLATFORM"),
        (f"{put_monthly:,}", "PUT / MONTH"),
        (f"{aic_monthly:,}", "AIC / MONTH"),
        (queue_text, "QUEUE DEPTH"),
    ]
    for i, (v, l) in enumerate(cells):
        cx = CARD_X + 22 + i * cell_w
        draw_stat_cell(draw, cx, stats_y, v, l)

    strip_y = stats_y + 80
    draw.text(
        (CARD_X + 22, strip_y),
        f"Scheduling lookahead: {lookahead_h}h   ·   "
        f"TikTok  ·  YouTube Shorts  ·  Instagram Reels  ·  Facebook Reels",
        font=F_SMALL,
        fill=TEXT_DIM,
    )

    statement = f"UPLOADM8 {name.upper()}"
    draw_footer(img, tax_code, statement, "subscription")
    return img


def _topup_unit_price(wallet: str, amount: int, topup_products: dict[str, dict[str, Any]]) -> float:
    key = f"uploadm8_{wallet}_{amount}"
    prod = topup_products.get(key)
    if prod:
        v = prod.get("price_usd", prod.get("price"))
        if v is not None:
            return float(v)
    fb = _CARD_TOPUP_PRICE_FALLBACK.get((wallet, amount))
    if fb is not None:
        return fb
    raise KeyError(
        f"No price for {wallet} pack {amount}: add TOPUP_PRODUCTS row or _CARD_TOPUP_PRICE_FALLBACK"
    )


def make_topup_card(
    wallet: str,
    amount: int,
    price: float,
    cloud_path: Path | None,
    tax_code: str = "txcd_10103001",
) -> Image.Image:
    wallet_u = wallet.upper()
    img = make_canvas()
    draw_card_body(img)
    accent = HEAD_ORANGE if wallet_u == "PUT" else HEAD_TEAL
    draw_header_band(img, accent)
    paste_logo(img, cloud_path)

    draw = ImageDraw.Draw(img)
    cx = CARD_X + CARD_W // 2
    y_amount = CARD_Y + HEADER_H + 35
    amount_str = f"{amount:,}"
    bbox = draw.textbbox((0, 0), amount_str, font=F_HUGE)
    tw = bbox[2] - bbox[0]
    draw.text((cx - tw // 2, y_amount), amount_str, font=F_HUGE, fill=WHITE)

    label = f"{wallet_u} TOKENS"
    bbox = draw.textbbox((0, 0), label, font=F_MED_BOLD)
    tw = bbox[2] - bbox[0]
    draw.text((cx - tw // 2, y_amount + 70), label, font=F_MED_BOLD, fill=accent)

    sub = "Upload Credits" if wallet_u == "PUT" else "AI Credits"
    bbox = draw.textbbox((0, 0), sub, font=F_MED)
    tw = bbox[2] - bbox[0]
    draw.text((cx - tw // 2, y_amount + 100), sub, font=F_MED, fill=TEXT_DIM)

    price_text = f"${price:.2f}"
    bbox = draw.textbbox((0, 0), price_text, font=F_PRICE)
    tw = bbox[2] - bbox[0]
    draw.text((cx - tw // 2, y_amount + 138), price_text, font=F_PRICE, fill=accent)

    if wallet_u == "PUT":
        vp = f"Publish {amount:,} videos across all connected platforms"
    else:
        vp = f"AI-powered optimization for {amount:,} uploads"
    bbox = draw.textbbox((0, 0), vp, font=F_SMALL)
    tw = bbox[2] - bbox[0]
    draw.text((cx - tw // 2, y_amount + 190), vp, font=F_SMALL, fill=TEXT_DIM)

    statement = f"UPLOADM8 {wallet_u} {amount}"
    draw_footer(img, tax_code, statement, "topup")
    return img


def _tier_filename_slug(ent_slug: str) -> str:
    if ent_slug == "free":
        return "starter"
    return ent_slug


def generate_all(
    tier_config: dict[str, dict[str, Any]],
    topup_products: dict[str, dict[str, Any]],
    out_dir: Path,
    cloud_icon: Path | None,
    *,
    public_card_slugs: tuple[str, ...] = _PUBLIC_CARD_SLUGS,
    topup_amounts: tuple[int, ...] = _TOPUP_AMOUNTS,
    log: Callable[[str], None] = print,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for slug in public_card_slugs:
        cfg = tier_config[slug]
        name = str(cfg["name"])
        price = float(cfg.get("price", 0))
        put_m = int(cfg.get("put_monthly", 0))
        aic_m = int(cfg.get("aic_monthly", 0))
        accts = int(cfg.get("max_accounts", 0))
        qd = int(cfg.get("queue_depth", 0))
        look = int(cfg.get("lookahead_hours", 0))
        trial_days = int(cfg.get("trial_days", 0) or 0)
        accent_orange = name == "Creator Pro"
        show_trial = trial_days > 0
        img = make_tier_card(
            name,
            price,
            put_m,
            aic_m,
            accts,
            qd,
            look,
            cloud_icon,
            accent_orange=accent_orange,
            show_trial_pill=show_trial,
        )
        fn = f"sub_{_tier_filename_slug(slug)}.png"
        dest = out_dir / fn
        img.save(dest, "PNG", optimize=True)
        log(f"  wrote {dest.name}")

    for amt in topup_amounts:
        price = _topup_unit_price("put", amt, topup_products)
        img = make_topup_card("PUT", amt, price, cloud_icon)
        dest = out_dir / f"topup_put_{amt}.png"
        img.save(dest, "PNG", optimize=True)
        log(f"  wrote {dest.name}")

    for amt in topup_amounts:
        price = _topup_unit_price("aic", amt, topup_products)
        img = make_topup_card("AIC", amt, price, cloud_icon)
        dest = out_dir / f"topup_aic_{amt}.png"
        img.save(dest, "PNG", optimize=True)
        log(f"  wrote {dest.name}")

    log(f"\nAll cards in {out_dir.resolve()}")
