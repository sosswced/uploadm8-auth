"""
UploadM8 Playwright Thumbnail Stage
=====================================
Render viral HTML/CSS thumbnail templates via headless Chromium.
Far superior to Pillow — enables:
  - Google Fonts (Bebas Neue, Inter, Montserrat Black)
  - CSS gradients, glassmorphism, neon glows
  - CSS filters (blur, brightness, contrast)
  - Animated-style static layouts
  - Drop shadows, border-radius, clip-path

Falls back to Pillow rendering if Playwright unavailable.

Templates (HTML/CSS):
  HEAT_HTML       — fire/dark cinematic
  NEON_HTML       — cyberpunk neon glow
  CINEMATIC_HTML  — letterbox film style
  CLEAN_HTML      — minimal gradient
  BRIGHT_HTML     — high-saturation pop
"""

import asyncio
import base64
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .context import JobContext
from .errors import SkipStage
from .outbound_rl import outbound_slot

logger = logging.getLogger("uploadm8-worker")

PLAYWRIGHT_ENABLED = os.environ.get("PLAYWRIGHT_ENABLED", "true").lower() == "true"

# Playwright availability
try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
    logger.info("[playwright] Playwright available")
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

# Singleton browser instance (shared across all renders in worker process)
_playwright_instance = None
_browser_instance    = None


async def get_browser():
    """Get or create the singleton Playwright browser."""
    global _playwright_instance, _browser_instance

    if _browser_instance:
        try:
            # Quick health check
            await _browser_instance.contexts()
            return _browser_instance
        except Exception:
            _browser_instance    = None
            _playwright_instance = None

    _playwright_instance = await async_playwright().start()
    _browser_instance    = await _playwright_instance.chromium.launch(
        headless=True,
        args=[
            "--no-sandbox",
            "--disable-setuid-sandbox",
            "--disable-dev-shm-usage",
            "--disable-gpu",
            "--disable-background-timer-throttling",
            "--disable-renderer-backgrounding",
        ],
    )
    logger.info("[playwright] Browser launched")
    return _browser_instance


async def render_html_thumbnail(
    html:    str,
    width:   int,
    height:  int,
    out_path: Path,
) -> Optional[Path]:
    """
    Render an HTML string to a screenshot saved at out_path.
    Returns out_path on success, None on failure.
    """
    if not PLAYWRIGHT_AVAILABLE or not PLAYWRIGHT_ENABLED:
        return None

    try:
        async with outbound_slot("playwright"):
            browser = await get_browser()
            context = await browser.new_context(viewport={"width": width, "height": height})
            page    = await context.new_page()

            try:
                await page.set_content(html, wait_until="load")
                # Wait for Google Fonts to load
                await asyncio.wait_for(
                    page.wait_for_function("() => document.fonts.ready"),
                    timeout=5.0,
                )
                screenshot = await page.screenshot(
                    type="jpeg",
                    quality=92,
                    omit_background=False,
                    full_page=False,
                )
                out_path.write_bytes(screenshot)
                logger.info(f"[playwright] Rendered {width}×{height} → {out_path.stat().st_size // 1024}KB")
                return out_path

            finally:
                await page.close()
                await context.close()

    except Exception as e:
        logger.warning(f"[playwright] Render error: {e}")
        return None


async def close_browser():
    """Gracefully close the Playwright browser. Call on worker shutdown."""
    global _browser_instance, _playwright_instance
    try:
        if _browser_instance:
            await _browser_instance.close()
        if _playwright_instance:
            await _playwright_instance.stop()
    except Exception:
        pass
    _browser_instance    = None
    _playwright_instance = None
    # Yield so subprocess transports finish on Windows (reduces closed-pipe noise in __del__)
    await asyncio.sleep(0)


# ── HTML Template Generators ──────────────────────────────────────────────────

def _b64_frame(frame_path: Path) -> str:
    """Base64-encode a frame for embedding in HTML data URI."""
    return base64.b64encode(frame_path.read_bytes()).decode()


def build_heat_html(
    frame_path: Path,
    headline:   str,
    subtext:    str,
    badge:      str,
    W: int, H: int,
) -> str:
    b64   = _b64_frame(frame_path)
    badge_html = f'<div class="badge">{badge}</div>' if badge else ""
    sub_html   = f'<div class="subtext">{subtext}</div>' if subtext else ""

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;700;900&display=swap" rel="stylesheet">
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ width:{W}px; height:{H}px; overflow:hidden; background:#000; font-family:'Inter',sans-serif; }}
.bg {{
  position:absolute; inset:0;
  background:url('data:image/jpeg;base64,{b64}') center/cover no-repeat;
  filter:brightness(0.45) contrast(1.2);
}}
.vignette {{
  position:absolute; inset:0;
  background:radial-gradient(ellipse at center, transparent 30%, rgba(0,0,0,0.85) 100%);
}}
.fire-bar-top    {{ position:absolute; top:0; left:0; right:0; height:6px; background:linear-gradient(90deg,#ff2200,#ff8800,#ff2200); }}
.fire-bar-bottom {{ position:absolute; bottom:0; left:0; right:0; height:10px; background:linear-gradient(90deg,#ff2200,#ff8800,#ff2200); }}
.content {{
  position:absolute; bottom:12%; left:5%; right:5%;
}}
.headline {{
  font-family:'Bebas Neue',sans-serif;
  font-size:{max(52, H//7)}px;
  color:#ffffff;
  line-height:1.0;
  text-shadow:4px 4px 0 #000, 6px 6px 12px rgba(0,0,0,0.8);
  text-transform:uppercase;
  letter-spacing:2px;
  word-break:break-word;
}}
.subtext {{
  font-family:'Inter',sans-serif;
  font-size:{max(22, H//22)}px;
  color:#ffaa40;
  font-weight:700;
  margin-top:8px;
  text-shadow:2px 2px 6px rgba(0,0,0,0.9);
}}
.badge {{
  position:absolute; top:4%; left:5%;
  background:#ff2200;
  color:#fff;
  font-family:'Inter',sans-serif;
  font-size:{max(18, H//28)}px;
  font-weight:900;
  padding:8px 16px;
  text-transform:uppercase;
  letter-spacing:1px;
  clip-path:polygon(0 0,calc(100% - 12px) 0,100% 50%,calc(100% - 12px) 100%,0 100%);
}}
</style>
</head>
<body>
<div class="bg"></div>
<div class="vignette"></div>
<div class="fire-bar-top"></div>
<div class="fire-bar-bottom"></div>
{badge_html}
<div class="content">
  <div class="headline">{headline}</div>
  {sub_html}
</div>
</body>
</html>"""


def build_neon_html(
    frame_path: Path,
    headline:   str,
    subtext:    str,
    badge:      str,
    W: int, H: int,
) -> str:
    b64     = _b64_frame(frame_path)
    sub_html   = f'<div class="subtext">{subtext}</div>' if subtext else ""
    badge_html = f'<div class="badge">{badge}</div>' if badge else ""

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@900&display=swap" rel="stylesheet">
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ width:{W}px; height:{H}px; overflow:hidden; background:#05050f; font-family:'Inter',sans-serif; }}
.bg {{
  position:absolute; inset:0;
  background:url('data:image/jpeg;base64,{b64}') center/cover no-repeat;
  filter:brightness(0.3) saturate(0.2);
}}
.neon-border {{
  position:absolute; inset:6px;
  border:2px solid #00ffdc;
  box-shadow:0 0 15px #00ffdc, inset 0 0 15px rgba(0,255,220,0.05);
}}
.mid-line {{
  position:absolute; top:50%; left:0; right:0; height:2px;
  background:linear-gradient(90deg, transparent, #ff00c8, transparent);
}}
.content {{
  position:absolute; bottom:8%; left:6%; right:50%;
}}
.headline {{
  font-family:'Bebas Neue',sans-serif;
  font-size:{max(52, H//7)}px;
  color:#ffee00;
  line-height:1.0;
  text-transform:uppercase;
  text-shadow:
    0 0 10px rgba(255,238,0,0.8),
    0 0 30px rgba(255,238,0,0.4),
    3px 3px 0 #000;
  word-break:break-word;
  letter-spacing:2px;
}}
.subtext {{
  font-family:'Inter',sans-serif;
  font-size:{max(20, H//24)}px;
  color:#00ffdc;
  font-weight:900;
  margin-top:10px;
  text-shadow:0 0 8px rgba(0,255,220,0.8);
}}
.badge {{
  position:absolute; top:4%; left:5%;
  background:#ff00c8;
  color:#fff;
  font-family:'Inter',sans-serif;
  font-size:{max(18, H//30)}px;
  font-weight:900;
  padding:8px 18px;
  text-transform:uppercase;
  box-shadow:0 0 20px rgba(255,0,200,0.6);
  letter-spacing:1px;
}}
</style>
</head>
<body>
<div class="bg"></div>
<div class="neon-border"></div>
<div class="mid-line"></div>
{badge_html}
<div class="content">
  <div class="headline">{headline}</div>
  {sub_html}
</div>
</body>
</html>"""


def build_cinematic_html(
    frame_path: Path,
    headline:   str,
    subtext:    str,
    badge:      str,
    W: int, H: int,
) -> str:
    b64      = _b64_frame(frame_path)
    bar_h    = int(H * 0.13)
    sub_html = f'<div class="subtext">{subtext}</div>' if subtext else ""

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;700&display=swap" rel="stylesheet">
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ width:{W}px; height:{H}px; overflow:hidden; background:#000; font-family:'Inter',sans-serif; }}
.bg {{
  position:absolute; inset:0;
  background:url('data:image/jpeg;base64,{b64}') center/cover no-repeat;
  filter:sepia(0.3) contrast(1.15) brightness(0.9);
}}
.warm-grade {{
  position:absolute; inset:0;
  background:rgba(255,150,50,0.08);
  mix-blend-mode:multiply;
}}
.bar-top    {{ position:absolute; top:0; left:0; right:0; height:{bar_h}px; background:#000; }}
.bar-bottom {{ position:absolute; bottom:0; left:0; right:0; height:{bar_h}px; background:#000; }}
.gold-line-top    {{ position:absolute; top:{bar_h}px; left:0; right:0; height:2px; background:linear-gradient(90deg,transparent,#d4af37,transparent); }}
.gold-line-bottom {{ position:absolute; bottom:{bar_h}px; left:0; right:0; height:2px; background:linear-gradient(90deg,transparent,#d4af37,transparent); }}
.bottom-text {{
  position:absolute; bottom:0; left:0; right:0; height:{bar_h}px;
  display:flex; align-items:center; justify-content:center;
}}
.headline {{
  font-family:'Bebas Neue',sans-serif;
  font-size:{max(44, bar_h // 2)}px;
  color:#fff;
  text-transform:uppercase;
  letter-spacing:4px;
  text-shadow:2px 2px 8px rgba(0,0,0,0.8);
}}
.top-text {{
  position:absolute; top:0; left:0; right:0; height:{bar_h}px;
  display:flex; align-items:center; justify-content:center;
}}
.subtext {{
  font-family:'Inter',sans-serif;
  font-size:{max(18, bar_h // 4)}px;
  color:#d4af37;
  text-transform:uppercase;
  letter-spacing:3px;
  font-weight:700;
}}
</style>
</head>
<body>
<div class="bg"></div>
<div class="warm-grade"></div>
<div class="bar-top"></div>
<div class="bar-bottom"></div>
<div class="gold-line-top"></div>
<div class="gold-line-bottom"></div>
<div class="top-text">
  <div class="subtext">{subtext or "&nbsp;"}</div>
</div>
<div class="bottom-text">
  <div class="headline">{headline}</div>
</div>
</body>
</html>"""


def build_bright_pop_html(
    frame_path: Path,
    headline:   str,
    subtext:    str,
    badge:      str,
    W: int, H: int,
) -> str:
    b64        = _b64_frame(frame_path)
    strip_h    = int(H * 0.28)
    sub_html   = f'<div class="subtext">{subtext}</div>' if subtext else ""
    badge_html = f'<div class="badge">{badge}</div>' if badge else ""

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@900&display=swap" rel="stylesheet">
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ width:{W}px; height:{H}px; overflow:hidden; background:#fff; font-family:'Inter',sans-serif; }}
.bg {{
  position:absolute; top:0; left:0; right:0; height:{H - strip_h}px;
  background:url('data:image/jpeg;base64,{b64}') center/cover no-repeat;
  filter:saturate(1.5) contrast(1.1);
}}
.yellow-strip {{
  position:absolute; bottom:0; left:0; right:0; height:{strip_h}px;
  background:#ffd700;
}}
.content {{
  position:absolute; bottom:0; left:0; right:0; height:{strip_h}px;
  padding:12px 5%;
}}
.headline {{
  font-family:'Bebas Neue',sans-serif;
  font-size:{max(50, strip_h // 2)}px;
  color:#111;
  line-height:1.0;
  text-transform:uppercase;
  letter-spacing:1px;
  text-shadow:3px 3px 0 rgba(0,0,0,0.15);
  word-break:break-word;
}}
.subtext {{
  font-family:'Inter',sans-serif;
  font-size:{max(20, strip_h // 6)}px;
  color:#444;
  font-weight:900;
  margin-top:4px;
}}
.badge {{
  position:absolute; top:4%; right:5%;
  background:#e60026;
  color:#fff;
  font-family:'Inter',sans-serif;
  font-size:{max(18, H//30)}px;
  font-weight:900;
  padding:8px 16px;
  text-transform:uppercase;
  border-radius:4px;
}}
</style>
</head>
<body>
<div class="bg"></div>
<div class="yellow-strip"></div>
{badge_html}
<div class="content">
  <div class="headline">{headline}</div>
  {sub_html}
</div>
</body>
</html>"""


def build_glitch_html(
    frame_path: Path,
    headline: str,
    subtext: str,
    badge: str,
    W: int, H: int,
) -> str:
    """High-impact glitch / RGB split headline (scroll-stopping)."""
    b64 = _b64_frame(frame_path)
    badge_html = f'<div class="badge">{badge}</div>' if badge else ""
    sub_html = f'<div class="subtext">{subtext}</div>' if subtext else ""
    fs = max(48, H // 7)
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@900&display=swap" rel="stylesheet">
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ width:{W}px; height:{H}px; overflow:hidden; background:#0a0a12; font-family:'Inter',sans-serif; }}
.bg {{
  position:absolute; inset:0;
  background:url('data:image/jpeg;base64,{b64}') center/cover no-repeat;
  filter:saturate(1.25) contrast(1.15);
}}
.scan {{
  position:absolute; inset:0;
  background:repeating-linear-gradient(0deg, rgba(255,255,255,0.03) 0px, transparent 2px, transparent 4px);
  pointer-events:none;
  mix-blend-mode:overlay;
}}
.headline-wrap {{
  position:absolute; bottom:10%; left:4%; right:4%;
}}
.glitch {{
  font-family:'Bebas Neue',sans-serif;
  font-size:{fs}px;
  line-height:0.95;
  text-transform:uppercase;
  color:#fff;
  position:relative;
}}
.glitch::before, .glitch::after {{
  content:attr(data-text);
  position:absolute; left:0; top:0; width:100%;
}}
.glitch::before {{
  animation:glitch1 0.4s infinite linear alternate-reverse;
  color:#ff00c8; clip-path:inset(0 0 55% 0);
}}
.glitch::after {{
  animation:glitch2 0.35s infinite linear alternate-reverse;
  color:#00fff7; clip-path:inset(45% 0 0 0);
}}
@keyframes glitch1 {{ 0%{{transform:translate(0,0)}} 100%{{transform:translate(-3px,1px)}} }}
@keyframes glitch2 {{ 0%{{transform:translate(0,0)}} 100%{{transform:translate(4px,-1px)}} }}
.subtext {{
  font-size:{max(18, H//28)}px; font-weight:900; color:#aef; margin-top:10px;
  text-shadow:0 0 8px #0ff;
}}
.badge {{
  position:absolute; top:4%; right:5%;
  background:#ff0044; color:#fff; font-weight:900; padding:8px 14px;
  font-size:{max(16, H//32)}px; text-transform:uppercase;
  box-shadow:0 0 20px #f04;
}}
</style>
</head>
<body>
<div class="bg"></div>
<div class="scan"></div>
{badge_html}
<div class="headline-wrap">
  <div class="glitch" data-text="{headline}">{headline}</div>
  {sub_html}
</div>
</body>
</html>"""


def build_chrome_html(
    frame_path: Path,
    headline: str,
    subtext: str,
    badge: str,
    W: int, H: int,
) -> str:
    """Metallic chrome-style headline."""
    b64 = _b64_frame(frame_path)
    badge_html = f'<div class="badge">{badge}</div>' if badge else ""
    sub_html = f'<div class="subtext">{subtext}</div>' if subtext else ""
    fs = max(46, H // 7)
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&display=swap" rel="stylesheet">
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ width:{W}px; height:{H}px; overflow:hidden; background:#111; font-family:'Bebas Neue',sans-serif; }}
.bg {{
  position:absolute; inset:0;
  background:url('data:image/jpeg;base64,{b64}') center/cover no-repeat;
  filter:brightness(0.35);
}}
.headline {{
  position:absolute; bottom:10%; left:5%; right:5%;
  font-size:{fs}px;
  line-height:1;
  text-transform:uppercase;
  background:linear-gradient(180deg,#fff 0%,#b0b8c8 40%,#6b7280 55%,#e8eef8 100%);
  -webkit-background-clip:text;
  -webkit-text-fill-color:transparent;
  background-clip:text;
  filter:drop-shadow(0 4px 0 #222) drop-shadow(0 8px 12px rgba(0,0,0,0.8));
}}
.subtext {{ font-size:{max(18, H//28)}px; color:#ddd; margin-top:12px; font-family:system-ui,sans-serif; font-weight:700; }}
.badge {{
  position:absolute; top:4%; left:5%;
  background:linear-gradient(135deg,#eee,#999); color:#111; font-weight:900;
  padding:8px 16px; font-size:{max(16, H//32)}px;
  font-family:system-ui,sans-serif;
}}
</style>
</head>
<body>
<div class="bg"></div>
{badge_html}
<div style="position:absolute;bottom:10%;left:5%;right:5%;">
  <div class="headline">{headline}</div>
  {sub_html}
</div>
</body>
</html>"""


def build_fire_scroll_html(
    frame_path: Path,
    headline: str,
    subtext: str,
    badge: str,
    W: int, H: int,
) -> str:
    """Fire / ember glow (distinct from HEAT bars)."""
    b64 = _b64_frame(frame_path)
    badge_html = f'<div class="badge">{badge}</div>' if badge else ""
    sub_html = f'<div class="subtext">{subtext}</div>' if subtext else ""
    fs = max(50, H // 7)
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@800&display=swap" rel="stylesheet">
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ width:{W}px; height:{H}px; overflow:hidden; background:#000; }}
.bg {{
  position:absolute; inset:0;
  background:url('data:image/jpeg;base64,{b64}') center/cover no-repeat;
  filter:brightness(0.4);
}}
.fire-glow {{
  position:absolute; inset:0;
  background:radial-gradient(ellipse 80% 100% at 50% 100%, rgba(255,80,0,0.55), transparent 55%);
  mix-blend-mode:screen;
}}
.headline {{
  position:absolute; bottom:8%; left:6%; right:6%;
  font-family:'Bebas Neue',sans-serif;
  font-size:{fs}px;
  line-height:1;
  text-transform:uppercase;
  color:#fff;
  text-shadow:
    0 0 20px #ff4400,
    0 0 40px #ff0000,
    0 0 80px #ff8800,
    4px 4px 0 #330000;
}}
.subtext {{ font-family:'Inter',sans-serif; font-size:{max(18, H//28)}px; color:#ffcc99; font-weight:800; margin-top:8px; }}
.badge {{
  position:absolute; top:5%; right:6%;
  background:#ff3d00; color:#fff; font-weight:900; padding:8px 14px;
  font-family:'Inter',sans-serif; font-size:{max(16, H//32)}px;
  box-shadow:0 0 24px #f50;
}}
</style>
</head>
<body>
<div class="bg"></div>
<div class="fire-glow"></div>
{badge_html}
<div class="headline">{headline}</div>
<div style="position:absolute;bottom:3%;left:6%;right:6%;">{sub_html}</div>
</body>
</html>"""


# Template dispatch
TEMPLATE_HTML_BUILDERS = {
    "HEAT":         build_heat_html,
    "NEON_DROP":    build_neon_html,
    "CINEMATIC":    build_cinematic_html,
    "BRIGHT_POP":   build_bright_pop_html,
    "GLITCH":       build_glitch_html,
    "CHROME":       build_chrome_html,
    "FIRE_SCROLL":  build_fire_scroll_html,
}


async def render_template(
    template:   str,
    frame_path: Path,
    headline:   str,
    subtext:    str,
    badge:      str,
    W: int, H: int,
    out_path: Path,
) -> Optional[Path]:
    """
    Render the appropriate HTML template via Playwright.
    Returns out_path on success, None if unavailable.
    """
    builder = TEMPLATE_HTML_BUILDERS.get(template)
    if not builder:
        return None

    html = builder(frame_path, headline, subtext, badge, W, H)
    return await render_html_thumbnail(html, W, H, out_path)
