"""Thumbnail URL resolution for upload list cards."""

from unittest.mock import patch

from services.upload.thumbnails import (
    card_thumbnail_url,
    thumbnail_storage_missing_flag,
    upload_card_thumbnail_href,
)


def test_card_thumbnail_prefers_youtube_cdn_over_proxy_when_r2_key_stale():
    upload_id = "ecffa18e-eb03-4270-91c6-989a88389629"
    platform_results = [
        {
            "platform": "youtube",
            "success": True,
            "platform_video_id": "dQw4w9WgXcQ",
        }
    ]
    with patch("services.upload.thumbnails.r2_object_exists", return_value=False):
        url = card_thumbnail_url(
            upload_id,
            thumbnail_r2_key=f"thumbnails/user/{upload_id}/thumbnail.jpg",
            output_artifacts={},
            platform_results=platform_results,
            upload_platforms=["youtube"],
        )
    assert url == "https://i.ytimg.com/vi/dQw4w9WgXcQ/hqdefault.jpg"
    assert url != upload_card_thumbnail_href(upload_id)


def test_card_thumbnail_uses_proxy_only_when_r2_object_exists():
    upload_id = "e0f77697-efbd-4bec-b8fb-fd853bb270c5"
    with patch("services.upload.thumbnails.r2_object_exists", return_value=True):
        url = card_thumbnail_url(
            upload_id,
            thumbnail_r2_key=f"thumbnails/user/{upload_id}/thumbnail.jpg",
            output_artifacts={},
            platform_results=[],
            upload_platforms=["youtube"],
        )
    assert url == upload_card_thumbnail_href(upload_id)


def test_card_thumbnail_no_proxy_when_r2_missing_and_no_platform_fallback():
    upload_id = "9d5ce8fe-93f5-473f-a940-27b56072240f"
    with patch("services.upload.thumbnails.r2_object_exists", return_value=False):
        url = card_thumbnail_url(
            upload_id,
            thumbnail_r2_key=f"thumbnails/user/{upload_id}/thumbnail.jpg",
            output_artifacts={},
            platform_results=[],
            upload_platforms=["youtube"],
        )
    assert url is None


def test_thumbnail_storage_missing_false_when_verified_proxy_used():
    upload_id = "fd9bccbf-10e5-486c-9c31-455e4fd7c445"
    proxy = upload_card_thumbnail_href(upload_id)
    sk = f"thumbnails/user/{upload_id}/thumbnail.jpg"
    assert not thumbnail_storage_missing_flag(
        primary_sk=sk,
        upload_id=upload_id,
        thumbnail_url=proxy,
        output_artifacts={},
        platform_results=[],
        upload_platforms=["youtube"],
    )


def test_thumbnail_storage_missing_still_flags_r2_repair_when_youtube_cdn_shown():
    upload_id = "35c84d56-a233-4295-935a-97f57f168494"
    yt = "https://i.ytimg.com/vi/abc123/hqdefault.jpg"
    sk = f"thumbnails/user/{upload_id}/thumbnail.jpg"
    assert thumbnail_storage_missing_flag(
        primary_sk=sk,
        upload_id=upload_id,
        thumbnail_url=yt,
        output_artifacts={},
        platform_results=[
            {"platform": "youtube", "success": True, "platform_video_id": "abc123"}
        ],
        upload_platforms=["youtube"],
    )
