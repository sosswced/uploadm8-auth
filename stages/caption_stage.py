"""
UploadM8 Caption Stage
======================
Generate titles and captions based on Trill data or video content.
Tier-gated feature.
"""

import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List

from .errors import CaptionError, SkipStage, ErrorCode
from .context import JobContext, CaptionResult


logger = logging.getLogger("uploadm8-worker")


# Trill-based caption templates
TRILL_CAPTIONS = {
    "gloryBoy": {
        "titles": [
            "GLORY BOY TOUR - Maximum Attack",
            "Glory Boy Mode Activated",
            "When the road calls, you answer",
            "Top speed unlocked",
            "Glory Boy Tour Episode"
        ],
        "captions": [
            "Speed is temporary. Glory is forever.",
            "No limits. Just vibes.",
            "When the road is clear and the car is ready.",
            "Glory Boy Tour continues.",
            "Send it with confidence."
        ],
        "hashtags": ["#GloryBoyTour", "#SpeedDemon", "#SendIt", "#CarContent", "#TrillScore"]
    },
    "euphoric": {
        "titles": [
            "Euphoric Drive",
            "Chasing euphoria",
            "That feeling when everything clicks",
            "Pure automotive bliss"
        ],
        "captions": [
            "When the drive hits different.",
            "This is why we do it.",
            "Euphoric state achieved.",
            "The perfect drive exists."
        ],
        "hashtags": ["#Euphoric", "#SpiritedDrive", "#CarLife", "#TrillScore"]
    },
    "sendIt": {
        "titles": [
            "Sending it",
            "Full send mode",
            "No hesitation",
            "Commit and execute"
        ],
        "captions": [
            "Sometimes you just gotta send it.",
            "Full commitment, no regrets.",
            "Send it Sunday vibes.",
            "The car wanted to play."
        ],
        "hashtags": ["#SendIt", "#SpiritedDrive", "#CarContent", "#TrillScore"]
    },
    "spirited": {
        "titles": [
            "Spirited drive",
            "Weekend warrior mode",
            "Good vibes only",
            "Taking the scenic route"
        ],
        "captions": [
            "A good drive clears the mind.",
            "Weekend drives hit different.",
            "The car needed exercise.",
            "Spirited, but responsible."
        ],
        "hashtags": ["#SpiritedDrive", "#WeekendWarrior", "#CarLife"]
    },
    "chill": {
        "titles": [
            "Cruising",
            "Relaxed vibes",
            "Easy Sunday drive",
            "Just enjoying the ride"
        ],
        "captions": [
            "Sometimes slow is the move.",
            "Enjoying the journey.",
            "Peaceful drives are underrated.",
            "Cruising vibes."
        ],
        "hashtags": ["#Cruising", "#RoadTrip", "#CarLife"]
    }
}


# Generic captions for videos without telemetry
GENERIC_CAPTIONS = {
    "titles": [
        "New upload",
        "Check this out",
        "Latest content",
        "Fresh drop"
    ],
    "captions": [
        "New content just dropped.",
        "Check out the latest.",
        "Here's something new.",
        "Fresh upload for you."
    ],
    "hashtags": ["#NewUpload", "#Content", "#UploadM8"]
}


def get_random_item(items: List[str], seed: str = "") -> str:
    """Get pseudo-random item based on seed."""
    import hashlib
    if not items:
        return ""
    # Use hash of seed to get consistent but varied selection
    h = int(hashlib.md5(seed.encode()).hexdigest(), 16)
    return items[h % len(items)]


def generate_trill_caption(ctx: JobContext) -> CaptionResult:
    """Generate caption based on Trill score."""
    if not ctx.trill:
        return CaptionResult(generated_by="manual")
    
    bucket = ctx.trill.bucket
    templates = TRILL_CAPTIONS.get(bucket, GENERIC_CAPTIONS)
    
    # Use upload_id as seed for consistent results
    seed = ctx.upload_id or str(datetime.now().timestamp())
    
    # Build title
    base_title = ctx.original_title or get_random_item(templates["titles"], seed + "title")
    title = f"{base_title}{ctx.trill.title_modifier}".strip()
    
    # Build caption
    base_caption = ctx.original_caption or get_random_item(templates["captions"], seed + "caption")
    
    # Add Trill score badge
    if ctx.trill.score >= 80:
        score_badge = f"Trill Score: {ctx.trill.score}/100"
        caption = f"{base_caption}\n\n{score_badge}"
    else:
        caption = base_caption
    
    # Combine hashtags
    hashtags = list(templates["hashtags"])
    if ctx.trill.hashtags:
        hashtags.extend(ctx.trill.hashtags)
    
    return CaptionResult(
        title=title,
        caption=caption,
        hashtags=list(dict.fromkeys(hashtags)),  # Dedupe
        generated_by="trill"
    )


def generate_generic_caption(ctx: JobContext) -> CaptionResult:
    """Generate generic caption for videos without telemetry."""
    seed = ctx.upload_id or str(datetime.now().timestamp())
    
    title = ctx.original_title or get_random_item(GENERIC_CAPTIONS["titles"], seed + "title")
    caption = ctx.original_caption or get_random_item(GENERIC_CAPTIONS["captions"], seed + "caption")
    
    return CaptionResult(
        title=title,
        caption=caption,
        hashtags=GENERIC_CAPTIONS["hashtags"],
        generated_by="generic"
    )


def generate_ai_caption(ctx: JobContext) -> CaptionResult:
    """
    Generate AI-powered caption based on video content.
    
    This is a placeholder for future AI integration.
    Could use vision models to analyze video frames.
    """
    # TODO: Implement AI caption generation
    # For now, fall back to trill or generic
    if ctx.trill:
        return generate_trill_caption(ctx)
    return generate_generic_caption(ctx)


async def run_caption_stage(ctx: JobContext) -> JobContext:
    """
    Execute caption generation stage.
    
    Args:
        ctx: Job context
        
    Returns:
        Updated context with caption data
        
    Raises:
        SkipStage: If user has manual title/caption and tier doesn't allow generation
    """
    # Check tier entitlements
    if not ctx.entitlements.can_generate_captions:
        if ctx.original_title and ctx.original_caption:
            # User provided both - use them
            ctx.caption = CaptionResult(
                title=ctx.original_title,
                caption=ctx.original_caption,
                hashtags=[],
                generated_by="manual"
            )
            raise SkipStage("User provided title/caption, tier doesn't allow generation")
        else:
            # Use whatever was provided
            ctx.caption = CaptionResult(
                title=ctx.original_title or ctx.filename,
                caption=ctx.original_caption or "",
                hashtags=[],
                generated_by="manual"
            )
            raise SkipStage("Caption generation not available for this tier")
    
    logger.info(f"Generating caption for upload {ctx.upload_id}")
    
    try:
        # Try AI caption if tier allows and no trill data
        if ctx.entitlements.can_use_ai_captions and not ctx.trill:
            ctx.caption = generate_ai_caption(ctx)
        # Use Trill-based caption if available
        elif ctx.trill:
            ctx.caption = generate_trill_caption(ctx)
        # Fall back to generic
        else:
            ctx.caption = generate_generic_caption(ctx)
        
        logger.info(f"Generated caption ({ctx.caption.generated_by}): {ctx.caption.title[:50]}...")
        return ctx
        
    except Exception as e:
        logger.error(f"Caption generation failed: {e}")
        # Don't fail the whole job - use original values
        ctx.caption = CaptionResult(
            title=ctx.original_title or ctx.filename,
            caption=ctx.original_caption or "",
            hashtags=[],
            generated_by="fallback"
        )
        return ctx
