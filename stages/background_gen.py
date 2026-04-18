"""
UploadM8 Background Generation
================================
Unified background removal + AI background generation utility.
Called by thumbnail_stage to create viral subject-on-AI-background composites.

Priority chain:
  1. rembg (free, local) — always attempted first
  2. remove.bg API ($0.09/image) — fallback if rembg fails + REMOVE_BG_ENABLED=true
  3. fal.ai FLUX (~$0.003/image) — AI background generation
  4. Replicate FLUX (~$0.012/image) — secondary AI background gen
  5. Pillow gradient — free fallback if all AI fails

Subject isolation + AI background = #1 viral thumbnail technique.
"""

import asyncio
import io
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Optional, Tuple

import httpx
import json
from PIL import Image

logger = logging.getLogger("uploadm8-worker")

_BG_NET_ERRS = (
    httpx.HTTPError,
    json.JSONDecodeError,
    KeyError,
    TypeError,
    ValueError,
    OSError,
)

# ── Config ────────────────────────────────────────────────────────────────────
REPLICATE_API_TOKEN  = os.environ.get("REPLICATE_API_TOKEN", "")
REPLICATE_ENABLED    = os.environ.get("REPLICATE_ENABLED", "true").lower() == "true"
FAL_KEY              = os.environ.get("FAL_KEY", "")
FAL_ENABLED          = os.environ.get("FAL_ENABLED", "true").lower() == "true"
REMOVE_BG_API_KEY    = os.environ.get("REMOVE_BG_API_KEY", "")
REMOVE_BG_ENABLED    = os.environ.get("REMOVE_BG_ENABLED", "false").lower() == "true"

# rembg availability
try:
    from rembg import remove as rembg_remove, new_session as rembg_new_session
    _rembg_session = rembg_new_session("isnet-general-use")
    REMBG_AVAILABLE = True
    logger.info("[bg] rembg available — local subject isolation enabled")
except ImportError:
    REMBG_AVAILABLE = False

_rembg_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rembg")


# ── Subject Isolation ─────────────────────────────────────────────────────────

async def isolate_subject(frame_path: Path, temp_dir: Path) -> Optional[Path]:
    """
    Remove background from best frame.
    Tries rembg (free) first, then remove.bg API as fallback.
    Returns path to RGBA PNG or None.
    """
    # Strategy 1: rembg (free, local)
    if REMBG_AVAILABLE:
        result = await _rembg_isolate(frame_path, temp_dir)
        if result:
            return result

    # Strategy 2: remove.bg API (paid, high quality)
    if REMOVE_BG_ENABLED and REMOVE_BG_API_KEY:
        result = await _removebg_api_isolate(frame_path, temp_dir)
        if result:
            return result

    logger.info("[bg] No subject isolation method available")
    return None


async def _rembg_isolate(frame_path: Path, temp_dir: Path) -> Optional[Path]:
    """Remove background using local rembg model."""
    out_path = temp_dir / "subject_nobg.png"
    try:
        image_bytes = frame_path.read_bytes()

        def _do_remove():
            result_bytes = rembg_remove(
                image_bytes,
                session=_rembg_session,
                force_return_bytes=True,
            )
            out_path.write_bytes(result_bytes)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(_rembg_executor, _do_remove)

        if out_path.exists() and out_path.stat().st_size > 1000:
            logger.info("[bg] rembg subject isolation ")
            return out_path

    except (OSError, PermissionError, ValueError, TypeError, RuntimeError, MemoryError) as e:
        logger.warning("[bg] rembg error: %s", e)

    return None


async def _removebg_api_isolate(frame_path: Path, temp_dir: Path) -> Optional[Path]:
    """Remove background using remove.bg API."""
    out_path = temp_dir / "subject_nobg_api.png"
    try:
        image_bytes = frame_path.read_bytes()

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                "https://api.remove.bg/v1.0/removebg",
                headers={"X-Api-Key": REMOVE_BG_API_KEY},
                files={"image_file": ("image.jpg", image_bytes, "image/jpeg")},
                data={"size": "auto", "type": "auto", "format": "png"},
            )

        if resp.status_code == 200:
            out_path.write_bytes(resp.content)
            credits_charged = resp.headers.get("X-Credits-Charged", "?")
            logger.info(f"[bg] remove.bg API  (credits charged: {credits_charged})")
            return out_path

        elif resp.status_code == 429:
            logger.warning("[bg] remove.bg rate limited")
        else:
            logger.warning(f"[bg] remove.bg error {resp.status_code}: {resp.text[:200]}")

    except asyncio.CancelledError:
        raise
    except _BG_NET_ERRS as e:
        logger.warning("[bg] remove.bg API error: %s", e)

    return None


# ── AI Background Generation ──────────────────────────────────────────────────

async def generate_ai_background(
    prompt: str,
    width:  int,
    height: int,
    temp_dir: Path,
) -> Optional[Path]:
    """
    Generate an AI background image using fal.ai (preferred) or Replicate.
    Returns path to downloaded JPEG or None.
    """
    # Strategy 1: fal.ai (cheaper, faster)
    if FAL_ENABLED and FAL_KEY:
        result = await _fal_generate(prompt, width, height, temp_dir)
        if result:
            return result

    # Strategy 2: Replicate FLUX
    if REPLICATE_ENABLED and REPLICATE_API_TOKEN:
        result = await _replicate_generate(prompt, width, height, temp_dir)
        if result:
            return result

    logger.info("[bg] No AI background generator available — using gradient fallback")
    return None


async def _fal_generate(prompt: str, width: int, height: int, temp_dir: Path) -> Optional[Path]:
    """Generate background via fal.ai FLUX schnell."""
    try:
        import fal_client

        # Determine aspect ratio
        aspect = _size_to_aspect(width, height)

        result = await fal_client.run_async(
            "fal-ai/flux/schnell",
            arguments={
                "prompt":               prompt,
                "image_size":           {"width": width, "height": height},
                "num_images":           1,
                "num_inference_steps":  4,
                "output_format":        "jpeg",
                "enable_safety_checker": False,
            },
        )

        images = result.get("images", [])
        if not images:
            return None

        image_url = images[0]["url"]
        return await _download_image(image_url, temp_dir / "ai_background_fal.jpg")

    except ImportError:
        logger.warning("[bg] fal-client not installed")
        return None
    except asyncio.CancelledError:
        raise
    except (TypeError, ValueError, KeyError, OSError, RuntimeError) as e:
        logger.warning("[bg] fal.ai error: %s", e)
        return None


async def _replicate_generate(prompt: str, width: int, height: int, temp_dir: Path) -> Optional[Path]:
    """Generate background via Replicate FLUX schnell."""
    try:
        import replicate

        aspect = _size_to_aspect(width, height)

        output = await replicate.async_run(
            "black-forest-labs/flux-schnell",
            input={
                "prompt":       prompt,
                "num_outputs":  1,
                "aspect_ratio": aspect,
                "output_format": "webp",
                "output_quality": 85,
            },
        )

        if not output:
            return None

        # output is a list of FileOutput objects
        image_url = output[0].url if hasattr(output[0], "url") else str(output[0])
        return await _download_image(image_url, temp_dir / "ai_background_replicate.jpg")

    except ImportError:
        logger.warning("[bg] replicate not installed")
        return None
    except asyncio.CancelledError:
        raise
    except (TypeError, ValueError, KeyError, OSError, RuntimeError) as e:
        logger.warning("[bg] Replicate error: %s", e)
        return None


async def _download_image(url: str, dest: Path) -> Optional[Path]:
    """Download image from URL to local path."""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.get(url)
        if resp.status_code == 200:
            dest.write_bytes(resp.content)
            logger.info(f"[bg] AI background downloaded: {dest.stat().st_size // 1024}KB")
            return dest
    except asyncio.CancelledError:
        raise
    except _BG_NET_ERRS as e:
        logger.warning("[bg] Download error: %s", e)
    return None


# ── Background Prompt Builder ─────────────────────────────────────────────────

def build_background_prompt(
    category:   str,
    mood:       str,
    emotion:    str,
    headline:   str,
    extra_context: str = "",
) -> str:
    """
    Generate a FLUX background prompt from content context.
    The prompt should describe a dramatic background — NOT a full scene.
    Subject will be composited on top via rembg.
    Optional extra_context: fused vision/geo/summary from pipeline (OpenAI + Flux).
    """
    category_prompts = {
        "automotive": "dramatic race track environment at golden hour, motion blur, asphalt, speed lines, cinematic",
        "sports_extreme": "intense stadium lighting, crowd bokeh, smoke effects, championship arena",
        "gaming": "neon cyberpunk city at night, holographic UI elements, digital grid, electric blue and purple",
        "music_performance": "concert stage with dramatic spotlights, fog machine, neon lights, crowd silhouette",
        "food_cooking": "moody dark kitchen with dramatic overhead light, steam, rustic wood textures",
        "travel_vlog": "stunning golden hour landscape, dramatic sky, cinematic travel destination",
        "fitness_workout": "gym aesthetic, motivational lighting, concrete and steel, dramatic shadows",
        "comedy_entertainment": "bright colorful backdrop, comic-book pop art, confetti, vivid colors",
        "educational": "clean minimalist gradient background, subtle geometric shapes, professional",
        "lifestyle_fashion": "clean studio background, gradient from deep to light, high fashion aesthetic",
        "nature_outdoors": "epic mountain vista at golden hour, dramatic clouds, cinematic color grade",
        "business_finance": "modern city skyline at night, glass buildings, professional corporate aesthetic",
        "other": "dramatic cinematic background, bokeh lights, deep colors, professional photography",
    }

    mood_modifiers = {
        "bold_dramatic":    "high contrast, dark vignette, cinematic grade, explosive energy",
        "neon_vibrant":     "neon glow effects, electric colors, cyberpunk aesthetic",
        "dark_cinematic":   "film noir, deep shadows, golden highlights, 35mm film grain",
        "clean_minimal":    "soft gradients, airy, minimal, professional studio",
        "bright_energetic": "vibrant saturated colors, energetic, high key lighting",
        "professional_clean": "corporate clean, neutral tones, subtle depth",
    }

    base = category_prompts.get(category, category_prompts["other"])
    mod  = mood_modifiers.get(mood, "cinematic, dramatic")

    prompt = f"{base}, {mod}, background only, no people, no text, ultra-detailed, photorealistic, 4K"
    extra = (extra_context or "").strip()
    if extra:
        prompt = f"{prompt}. Scene alignment: {extra[:500]}"
    return prompt


# ── Composite Subject onto Background ────────────────────────────────────────

def composite_subject_on_background(
    subject_path:    Path,
    background_path: Path,
    canvas_w:        int,
    canvas_h:        int,
    anchor:          str = "right",
    scale:           float = 0.82,
) -> Image.Image:
    """
    Composite a rembg-extracted subject (RGBA) onto a background image.
    Returns a composite PIL Image.
    """
    # Load and resize background to canvas
    bg = Image.open(background_path).convert("RGB")
    bg = _fill_canvas(bg, canvas_w, canvas_h)

    # Load and scale subject
    subject = Image.open(subject_path).convert("RGBA")
    target_h = int(canvas_h * scale)
    ratio    = target_h / subject.height
    target_w = int(subject.width * ratio)
    subject  = subject.resize((target_w, target_h), Image.LANCZOS)

    # Position subject
    if anchor == "right":
        x = canvas_w - target_w
        y = canvas_h - target_h
    elif anchor == "center":
        x = (canvas_w - target_w) // 2
        y = canvas_h - target_h
    else:
        x = 0
        y = canvas_h - target_h

    # Composite
    composite = bg.copy().convert("RGBA")
    composite.paste(subject, (x, y), subject)
    return composite.convert("RGB")


def _fill_canvas(img: Image.Image, W: int, H: int) -> Image.Image:
    """Resize + centre-crop to W×H."""
    iw, ih = img.size
    scale  = max(W / iw, H / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    img    = img.resize((nw, nh), Image.LANCZOS)
    x0, y0 = (nw - W) // 2, (nh - H) // 2
    return img.crop((x0, y0, x0 + W, y0 + H))


def _size_to_aspect(w: int, h: int) -> str:
    ratio = w / h
    if abs(ratio - 16 / 9) < 0.1:  return "16:9"
    if abs(ratio - 9 / 16) < 0.1:  return "9:16"
    if abs(ratio - 1.0) < 0.1:     return "1:1"
    if abs(ratio - 4 / 3) < 0.1:   return "4:3"
    return "1:1"


# ── Optional Replicate: text-forward hero images + ControlNet / Kontext ─────

REPLICATE_TEXT_IMAGE_MODEL = os.environ.get("REPLICATE_TEXT_IMAGE_MODEL", "").strip()
REPLICATE_CONTROLNET_MODEL = os.environ.get("REPLICATE_CONTROLNET_MODEL", "").strip()
REPLICATE_KONTEXT_MODEL = os.environ.get("REPLICATE_KONTEXT_MODEL", "").strip()


async def generate_text_hero_image_replicate(
    prompt: str,
    width: int,
    height: int,
    temp_dir: Path,
) -> Optional[Path]:
    """
    Text-heavy hero still (Ideogram-class) when REPLICATE_TEXT_IMAGE_MODEL is set
    (e.g. ideogram-ai/ideogram-v2 — check Replicate model card for input schema).
    """
    if not REPLICATE_TEXT_IMAGE_MODEL or not REPLICATE_ENABLED or not REPLICATE_API_TOKEN:
        return None
    try:
        import replicate

        aspect = _size_to_aspect(width, height)
        out = await replicate.async_run(
            REPLICATE_TEXT_IMAGE_MODEL,
            input={
                "prompt": prompt[:900],
                "aspect_ratio": aspect,
            },
        )
        if not out:
            return None
        image_url = out[0].url if hasattr(out[0], "url") else str(out[0])
        return await _download_image(image_url, temp_dir / "text_hero_replicate.jpg")
    except asyncio.CancelledError:
        raise
    except (TypeError, ValueError, KeyError, OSError, RuntimeError) as e:
        logger.warning("[bg] text hero replicate error: %s", e)
        return None


async def generate_controlnet_background_replicate(
    prompt: str,
    control_image_path: Path,
    width: int,
    height: int,
    temp_dir: Path,
) -> Optional[Path]:
    """
    Generic ControlNet-style pass-through: model id must match Replicate schema.
    Set REPLICATE_CONTROLNET_MODEL to a model that accepts prompt + control image.
    """
    if not REPLICATE_CONTROLNET_MODEL or not REPLICATE_ENABLED or not REPLICATE_API_TOKEN:
        return None
    if not control_image_path.exists():
        return None
    try:
        import replicate

        with open(control_image_path, "rb") as fh:
            out = await replicate.async_run(
                REPLICATE_CONTROLNET_MODEL,
                input={
                    "prompt": prompt[:900],
                    "image": fh,
                    "width": width,
                    "height": height,
                },
            )
        if not out:
            return None
        image_url = out[0].url if hasattr(out[0], "url") else str(out[0])
        return await _download_image(image_url, temp_dir / "controlnet_bg_replicate.jpg")
    except asyncio.CancelledError:
        raise
    except (TypeError, ValueError, KeyError, OSError, RuntimeError) as e:
        logger.warning("[bg] controlnet replicate error: %s", e)
        return None


async def generate_kontext_background_replicate(
    frame_path: Path,
    prompt: str,
    width: int,
    height: int,
    temp_dir: Path,
) -> Optional[Path]:
    """
    FLUX Kontext–style reference image → new scene (set REPLICATE_KONTEXT_MODEL).
    Useful when you want the model to respect framing while changing the backdrop.
    """
    if not REPLICATE_KONTEXT_MODEL or not REPLICATE_ENABLED or not REPLICATE_API_TOKEN:
        return None
    if not frame_path.exists():
        return None
    try:
        import replicate

        with open(frame_path, "rb") as fh:
            out = await replicate.async_run(
                REPLICATE_KONTEXT_MODEL,
                input={
                    "prompt": prompt[:900],
                    "image": fh,
                    "aspect_ratio": _size_to_aspect(width, height),
                },
            )
        if not out:
            return None
        image_url = out[0].url if hasattr(out[0], "url") else str(out[0])
        return await _download_image(image_url, temp_dir / "kontext_bg_replicate.jpg")
    except asyncio.CancelledError:
        raise
    except (TypeError, ValueError, KeyError, OSError, RuntimeError) as e:
        logger.warning("[bg] kontext replicate error: %s", e)
        return None
