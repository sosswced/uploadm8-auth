"""
UploadM8 Twelve Labs Stage
==========================
Deep video understanding via Twelve Labs Pegasus 1.2 model.
Goes far beyond frame-by-frame GPT vision — understands the full
narrative arc, action sequences, and scene context of the entire video.

Output: Rich scene description injected into caption + thumbnail prompts.

Free tier: 600 cumulative minutes. Then $0.12/min for Pegasus.
Runs when the user enables Scene Understanding in upload preferences and
``TWELVE_LABS_API_KEY`` is set.

Flow:
  1. Check if video is already indexed (cache by upload_id)
  2. If not, create index + upload video → wait for indexing
  3. Run generate task: "Describe this video in detail for social media captions"
  4. Store result in ctx.video_understanding
"""

import asyncio
import json
import logging
import os
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from .context import JobContext
from .errors import SkipStage
from .ai_service_costs import user_pref_ai_service_enabled
from services.provider_error_trace import append_provider_error

logger = logging.getLogger("uploadm8-worker")

TWELVE_LABS_API_KEY = os.environ.get("TWELVE_LABS_API_KEY", "")
TWELVELABS_INDEX_ID = os.environ.get("TWELVELABS_INDEX_ID", "")  # Pre-created index

# When Video Intelligence already produced rich object/text/logo/person tracks
# Twelve Labs is largely redundant — every TL call costs ~$0.10/min indexed.
# This flag lets ops auto-skip TL for clips that already have evidence floor.
# Default ON (skip TL when VI gives us enough). Set to "false" to keep both
# services running every time.
TWELVELABS_SKIP_WHEN_VI_RICH = (
    os.environ.get("TWELVELABS_SKIP_WHEN_VI_RICH", "true").lower() == "true"
)
TWELVELABS_VI_RICH_OBJECT_MIN = int(os.environ.get("TWELVELABS_VI_RICH_OBJECT_MIN", "3") or 3)
TWELVELABS_VI_RICH_LOGO_MIN = int(os.environ.get("TWELVELABS_VI_RICH_LOGO_MIN", "1") or 1)

# Optional comma-separated custom prompts. Each one runs a single
# /analyze call and stores the returned text under
# ``video_understanding["custom_queries"][query_label]``. Disabled when blank.
# Example:
#   TWELVELABS_CUSTOM_QUERIES="brands_visible:list every brand or product visible|memorable_quote:transcribe the single most memorable quote spoken"
TWELVELABS_CUSTOM_QUERIES = (os.environ.get("TWELVELABS_CUSTOM_QUERIES") or "").strip()
TL_BASE_URL          = "https://api.twelvelabs.io/v1.3"
INDEX_POLL_INTERVAL  = float(os.environ.get("TWELVELABS_POLL_INTERVAL_SEC", "10"))  # seconds
INDEX_MAX_POLLS      = int(os.environ.get("TWELVELABS_MAX_POLLS", "30"))            # legacy floor
INDEX_MAX_WAIT_SEC   = int(os.environ.get("TWELVELABS_MAX_WAIT_SEC", "1800"))       # 30m cap


def _effective_poll_budget(file_size_mb: float) -> int:
    """
    Adaptive task polling budget.
    The fixed 5m window was too tight for larger clips and busy API windows.
    """
    interval = max(1.0, float(INDEX_POLL_INTERVAL))
    # Baseline 5m + 90s per ~25MB, capped by TWELVELABS_MAX_WAIT_SEC (default 30m)
    dynamic_wait = 300 + int(max(0.0, file_size_mb) / 25.0) * 90
    wait_cap = max(300, int(INDEX_MAX_WAIT_SEC))
    wait_sec = min(wait_cap, dynamic_wait)
    polls = max(int(INDEX_MAX_POLLS), int(math.ceil(wait_sec / interval)))
    return max(1, polls)


async def run_twelvelabs_stage(ctx: JobContext) -> JobContext:
    """
    Index and analyze video with Twelve Labs.
    Stores rich scene description in ctx.video_understanding.
    Non-fatal — skipped gracefully if disabled or API errors.
    """
    ctx.mark_stage("twelvelabs")

    tier_allowed = getattr(ctx.entitlements, "allowed_ai_services", None) if ctx.entitlements else None
    tier_allowed_set = set(tier_allowed) if tier_allowed is not None else None
    if not user_pref_ai_service_enabled(
        ctx.user_settings or {}, "twelvelabs", default=False, allowed_services=tier_allowed_set
    ):
        raise SkipStage("Scene Understanding disabled in upload preferences (aiServiceSceneUnderstanding)")

    if not TWELVE_LABS_API_KEY:
        raise SkipStage("TWELVE_LABS_API_KEY not configured")

    # Cost gate: if Video Intelligence already produced rich tracks the
    # additional TL spend rarely improves caption accuracy. Skip unless the
    # user explicitly forces TL, Vision labels are weak (depth router), or
    # this gate is disabled by env.
    if TWELVELABS_SKIP_WHEN_VI_RICH:
        vi = getattr(ctx, "video_intelligence", None) or getattr(
            ctx, "video_intelligence_context", None
        ) or {}
        if isinstance(vi, dict) and not vi.get("error"):
            n_obj = len(vi.get("object_tracks") or [])
            n_logo = len(vi.get("logos") or [])
            us = ctx.user_settings or {}
            force_tl = bool(us.get("force_twelvelabs") or us.get("forceTwelveLabs"))
            vision_weak = False
            try:
                from core.vision_labels import vision_labels_are_weak

                vc = getattr(ctx, "vision_context", None) or {}
                if isinstance(vc, dict) and vc:
                    vision_weak = vision_labels_are_weak(
                        vc.get("label_names") or [],
                        landmark_names=vc.get("landmark_names") or [],
                        logo_names=vc.get("logo_names") or [],
                        ocr_text=str(vc.get("ocr_text") or ""),
                    )
            except Exception:
                vision_weak = False
            if (
                (n_obj >= TWELVELABS_VI_RICH_OBJECT_MIN or n_logo >= TWELVELABS_VI_RICH_LOGO_MIN)
                and not force_tl
                and not vision_weak
            ):
                raise SkipStage(
                    f"Video Intelligence already rich (objects={n_obj}, logos={n_logo}) — "
                    "skipping Twelve Labs to control cost. "
                    "Set forceTwelveLabs=true, enable depth router, or "
                    "TWELVELABS_SKIP_WHEN_VI_RICH=false to override."
                )

    video_path = None
    for candidate in (ctx.processed_video_path, ctx.local_video_path):
        if candidate and Path(candidate).exists():
            video_path = Path(candidate)
            break
    if not video_path:
        raise SkipStage("No local video file for Twelve Labs")

    try:
        index_id = TWELVELABS_INDEX_ID or await _get_or_create_index(ctx=ctx)
        if not index_id:
            raise SkipStage("Could not get/create Twelve Labs index")

        # Upload and index the video
        video_id = await _upload_and_index(video_path, index_id, ctx.upload_id, ctx=ctx)
        if not video_id:
            raise SkipStage("Video indexing failed")

        # Generate scene description
        description = await _generate_description(video_id, ctx=ctx)
        if not description:
            raise SkipStage("Description generation failed")

        # Generate title suggestion
        title_suggestion = await _generate_title(video_id, ctx=ctx)

        ctx.video_understanding = {
            "scene_description": description,
            "title_suggestion":  title_suggestion or "",
            "video_id":          video_id,
            "index_id":          index_id,
        }

        # Custom queries — domain-specific prompts (brands, quotes, locations,
        # etc.) parsed from TWELVELABS_CUSTOM_QUERIES. Each result is stored
        # under ``video_understanding["custom_queries"][label]`` and surfaced
        # to the M8 prompt + hydration enforcer via build_scene_graph.
        if TWELVELABS_CUSTOM_QUERIES:
            cq_specs = _parse_custom_queries(TWELVELABS_CUSTOM_QUERIES)
            if cq_specs:
                cq_results: Dict[str, str] = {}
                for label, prompt_text in cq_specs:
                    try:
                        ans = await _generate_custom(video_id, prompt_text, ctx=ctx)
                        if ans:
                            cq_results[label] = ans
                    except Exception as cq_e:
                        logger.warning(
                            "[twelvelabs] custom query %r failed: %s", label, cq_e
                        )
                if cq_results:
                    ctx.video_understanding["custom_queries"] = cq_results
                    logger.info(
                        "[twelvelabs] custom_queries succeeded: %s",
                        ",".join(cq_results.keys()),
                    )

        logger.info(
            f"[twelvelabs]  video_id={video_id} "
            f"description_len={len(description)} chars"
        )
        return ctx

    except asyncio.CancelledError:
        raise
    except SkipStage:
        raise
    except (
        httpx.RequestError,
        httpx.HTTPError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
        OSError,
    ) as e:
        logger.warning("[twelvelabs] Non-fatal error: %s", e)
        append_provider_error(
            ctx,
            provider="twelvelabs",
            stage="twelvelabs_stage",
            operation="index_or_generate",
            message=str(e),
            exception_type=type(e).__name__,
        )
        ctx.video_understanding = {}
        return ctx


async def _get_or_create_index(*, ctx: Optional[JobContext] = None) -> Optional[str]:
    """Get existing uploadm8 index or create one with Pegasus + Marengo."""
    headers = {"x-api-key": TWELVE_LABS_API_KEY, "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Check for existing uploadm8 index
        resp = await client.get(f"{TL_BASE_URL}/indexes", headers=headers)
        if resp.status_code == 200:
            indexes = resp.json().get("data", [])
            for idx in indexes:
                if "uploadm8" in idx.get("name", "").lower():
                    logger.info(f"[twelvelabs] Using existing index: {idx['_id']}")
                    return idx["_id"]

        # Create new index
        resp = await client.post(
            f"{TL_BASE_URL}/indexes",
            headers=headers,
            json={
                "name": "uploadm8-content",
                "models": [
                    {"name": "pegasus1.2",  "options": ["visual", "audio"]},
                    {"name": "marengo3.0",  "options": ["visual", "audio"]},
                ],
            },
        )

        if resp.status_code in (200, 201):
            index_id = resp.json().get("_id") or resp.json().get("id")
            logger.info(f"[twelvelabs] Created index: {index_id}")
            return index_id

        logger.warning(f"[twelvelabs] Index creation failed: {resp.status_code} {resp.text[:200]}")
        if ctx is not None:
            append_provider_error(
                ctx,
                provider="twelvelabs",
                stage="twelvelabs_stage",
                operation="create_index",
                message="index creation failed",
                http_status=resp.status_code,
                response_body_snippet=resp.text[:1200],
            )
        return None


async def _upload_and_index(
    video_path: Path,
    index_id: str,
    upload_id: str,
    *,
    ctx: Optional[JobContext] = None,
) -> Optional[str]:
    """Upload video file to Twelve Labs and wait for indexing to complete."""
    headers = {"x-api-key": TWELVE_LABS_API_KEY}

    file_size_mb = video_path.stat().st_size / 1024 / 1024
    logger.info(f"[twelvelabs] Uploading {file_size_mb:.1f}MB video for indexing...")

    poll_budget = _effective_poll_budget(file_size_mb)
    max_wait_min = (poll_budget * INDEX_POLL_INTERVAL) / 60.0
    logger.info(
        "[twelvelabs] Poll budget: up to %d checks every %.1fs (~%.1f min max wait)",
        poll_budget,
        INDEX_POLL_INTERVAL,
        max_wait_min,
    )

    async with httpx.AsyncClient(timeout=300.0) as client:
        with open(video_path, "rb") as f:
            resp = await client.post(
                f"{TL_BASE_URL}/tasks",
                headers=headers,
                data={
                    "index_id":      index_id,
                    "language":      "en",
                    "enable_video_stream": "false",
                },
                files={"video_file": (video_path.name, f, "video/mp4")},
            )

        if resp.status_code not in (200, 201):
            logger.warning(f"[twelvelabs] Upload failed: {resp.status_code} {resp.text[:300]}")
            if ctx is not None:
                append_provider_error(
                    ctx,
                    provider="twelvelabs",
                    stage="twelvelabs_stage",
                    operation="upload_task",
                    message="task upload failed",
                    http_status=resp.status_code,
                    response_body_snippet=resp.text[:1200],
                )
            return None

        task_id = resp.json().get("_id") or resp.json().get("id")
        if not task_id:
            logger.warning("[twelvelabs] No task_id in upload response")
            return None

        logger.info(f"[twelvelabs] Task created: {task_id} — polling for completion...")

        # Poll task status
        last_status = ""
        for attempt in range(poll_budget):
            await asyncio.sleep(INDEX_POLL_INTERVAL)

            status_resp = await client.get(
                f"{TL_BASE_URL}/tasks/{task_id}",
                headers=headers,
            )

            if status_resp.status_code != 200:
                if (attempt + 1) % 6 == 0:
                    logger.warning(
                        "[twelvelabs] Task poll non-200 (%s) on attempt %d/%d",
                        status_resp.status_code,
                        attempt + 1,
                        poll_budget,
                    )
                continue

            data   = status_resp.json()
            status = data.get("status", "")
            if status != last_status:
                logger.info(
                    "[twelvelabs] Task %s status=%s (%d/%d)",
                    task_id,
                    status or "unknown",
                    attempt + 1,
                    poll_budget,
                )
                last_status = status

            if status == "ready":
                video_id = data.get("video_id")
                logger.info(f"[twelvelabs] Indexing complete — video_id={video_id}")
                return video_id

            elif status in ("failed", "error"):
                logger.warning(f"[twelvelabs] Indexing failed: {data}")
                if ctx is not None:
                    append_provider_error(
                        ctx,
                        provider="twelvelabs",
                        stage="twelvelabs_stage",
                        operation="poll_task",
                        message=f"task status={status}",
                        response_body_snippet=json.dumps(data, default=str)[:1200],
                    )
                return None

            logger.debug(f"[twelvelabs] Polling attempt {attempt + 1}/{poll_budget}: status={status}")

        logger.warning(
            "[twelvelabs] Indexing timed out after %d checks (~%.1f min)",
            poll_budget,
            (poll_budget * INDEX_POLL_INTERVAL) / 60.0,
        )
        return None


async def _generate_description(video_id: str, *, ctx: Optional[JobContext] = None) -> Optional[str]:
    """Generate a detailed scene description using Pegasus generate endpoint."""
    headers = {"x-api-key": TWELVE_LABS_API_KEY, "Content-Type": "application/json"}

    prompt = (
        "Describe this video in detail for social media caption generation. "
        "Include: the main subject/person, what they are doing, the location/environment, "
        "any notable actions or moments, the overall mood and energy, "
        "and any text or signs visible. Be specific and vivid."
    )

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{TL_BASE_URL}/analyze",
            headers=headers,
            json={
                "video_id": video_id,
                "prompt":   prompt,
                "temperature": 0.3,
            },
        )

        if resp.status_code == 200:
            return _extract_analyze_text(resp.text)

        logger.warning(f"[twelvelabs] Generate failed: {resp.status_code} {resp.text[:200]}")
        if ctx is not None:
            append_provider_error(
                ctx,
                provider="twelvelabs",
                stage="twelvelabs_stage",
                operation="generate_description",
                message="analyze failed",
                http_status=resp.status_code,
                response_body_snippet=resp.text[:1200],
            )
        return None


async def _generate_title(video_id: str, *, ctx: Optional[JobContext] = None) -> Optional[str]:
    """Generate a viral title suggestion."""
    headers = {"x-api-key": TWELVE_LABS_API_KEY, "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{TL_BASE_URL}/analyze",
            headers=headers,
            json={
                "video_id": video_id,
                "prompt":   "Generate one punchy, viral social media title for this video. Max 10 words. No hashtags.",
                "temperature": 0.7,
            },
        )

        if resp.status_code == 200:
            return _extract_analyze_text(resp.text).strip()

        if ctx is not None:
            append_provider_error(
                ctx,
                provider="twelvelabs",
                stage="twelvelabs_stage",
                operation="generate_title",
                message="analyze title failed",
                http_status=resp.status_code,
                response_body_snippet=resp.text[:1200],
            )
        return None


def _parse_custom_queries(env_value: str) -> List[tuple]:
    """Parse ``label1:prompt1|label2:prompt2`` env spec into [(label, prompt), ...].

    Labels are slugified: lowercase, alphanumeric + underscore. Empty
    labels and malformed pairs are silently dropped. Useful for niche-
    specific prompts the user wants on every clip (e.g. "list_brands",
    "memorable_quote", "location_clue").
    """
    out: List[tuple] = []
    for chunk in env_value.split("|"):
        chunk = chunk.strip()
        if not chunk or ":" not in chunk:
            continue
        label, _, prompt = chunk.partition(":")
        label_slug = "".join(
            ch if ch.isalnum() or ch == "_" else "_"
            for ch in label.strip().lower()
        )
        prompt = prompt.strip()
        if not label_slug or not prompt:
            continue
        out.append((label_slug, prompt))
    return out[:6]


async def _generate_custom(video_id: str, prompt_text: str, *, ctx: Optional[JobContext] = None) -> Optional[str]:
    """Run a single Twelve Labs analyze with a user-supplied prompt."""
    if not prompt_text or not video_id:
        return None
    headers = {"x-api-key": TWELVE_LABS_API_KEY, "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{TL_BASE_URL}/analyze",
            headers=headers,
            json={
                "video_id": video_id,
                "prompt": prompt_text,
                "temperature": 0.2,
            },
        )
        if resp.status_code == 200:
            txt = _extract_analyze_text(resp.text).strip()
            return txt or None
        logger.warning("[twelvelabs] custom analyze failed %s: %s", resp.status_code, resp.text[:200])
        if ctx is not None:
            append_provider_error(
                ctx,
                provider="twelvelabs",
                stage="twelvelabs_stage",
                operation="generate_custom",
                message=f"custom analyze failed: {prompt_text[:120]}",
                http_status=resp.status_code,
                response_body_snippet=resp.text[:1200],
            )
        return None


def _extract_analyze_text(raw_body: str) -> str:
    """
    Twelve Labs /analyze currently returns NDJSON streaming events even on non-stream requests.
    Parse both regular JSON objects and event-stream NDJSON payloads.
    """
    # Standard JSON response fallback (future-proof if API changes back).
    try:
        data = json.loads(raw_body)
        if isinstance(data, dict):
            return (data.get("data") or data.get("text") or "").strip()
    except json.JSONDecodeError:
        pass

    chunks: List[str] = []
    for line in raw_body.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and event.get("event_type") == "text_generation":
            text = event.get("text")
            if isinstance(text, str):
                chunks.append(text)

    return "".join(chunks).strip()
