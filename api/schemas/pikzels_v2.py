"""Pydantic bodies for Pikzels v2 proxy routes (thumbnail studio + admin)."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class PikzelsV2TelemetryFields(BaseModel):
    """Optional linkage for ML / variant ranking (Thumbnail Studio → uploads)."""
    studio_variant_id: Optional[str] = Field(default=None, max_length=128)
    upload_id: Optional[str] = Field(default=None, max_length=64)


class PikzelsV2PromptBody(PikzelsV2TelemetryFields):
    prompt: str = Field(min_length=1, max_length=1000)
    model: str = "pkz_4"
    format: str = "16:9"
    support_image_url: Optional[str] = None
    support_image_base64: Optional[str] = None
    persona: Optional[str] = None
    style: Optional[str] = None


class PikzelsV2RecreateBody(PikzelsV2TelemetryFields):
    """Pikzels v2 create-from-image (Recreate™)."""
    prompt: Optional[str] = Field(default=None, max_length=1000)
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    support_image_url: Optional[str] = None
    support_image_base64: Optional[str] = None
    image_weight: Optional[str] = "medium"
    model: str = "pkz_4"
    format: str = "16:9"
    persona: Optional[str] = None
    style: Optional[str] = None


class PikzelsV2EditBody(PikzelsV2TelemetryFields):
    """Pikzels v2 edit (Edit + One-Click Fix™ share this endpoint)."""
    prompt: str = Field(min_length=1, max_length=1000)
    format: str = "16:9"
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    mask_url: Optional[str] = None
    mask_base64: Optional[str] = None
    support_image_url: Optional[str] = None
    support_image_base64: Optional[str] = None


class PikzelsV2FaceswapBody(PikzelsV2TelemetryFields):
    format: str = "16:9"
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    face_image: Optional[str] = None
    face_image_base64: Optional[str] = None
    mask_url: Optional[str] = None
    mask_base64: Optional[str] = None


class PikzelsV2ScoreBody(PikzelsV2TelemetryFields):
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    title: Optional[str] = Field(default=None, max_length=200)


class PikzelsV2TitlesBody(PikzelsV2TelemetryFields):
    prompt: Optional[str] = Field(default=None, max_length=2000)
    support_image_url: Optional[str] = None
    support_image_base64: Optional[str] = None


class PikzelsV2PikzonalityBody(PikzelsV2TelemetryFields):
    name: str = Field(min_length=1, max_length=25)
    image_urls: Optional[List[str]] = None
    image_base64s: Optional[List[str]] = None
