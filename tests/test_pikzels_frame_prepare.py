"""Pikzels source-frame JPEG preparation."""

from pathlib import Path

from PIL import Image

from stages.pikzels_api import _jpeg_bytes_for_pikzels_frame


def test_jpeg_bytes_for_pikzels_frame_reencodes_png(tmp_path: Path):
    src = tmp_path / "frame.png"
    Image.new("RGB", (640, 360), color=(20, 40, 60)).save(src, format="PNG")
    out = _jpeg_bytes_for_pikzels_frame(src)
    assert out[:2] == b"\xff\xd8"  # JPEG SOI
    from io import BytesIO

    with Image.open(BytesIO(out)) as im:
        assert im.size == (640, 360)
        assert im.format == "JPEG"


def test_jpeg_bytes_rejects_tiny_frame(tmp_path: Path):
    src = tmp_path / "tiny.png"
    Image.new("RGB", (16, 16), color=(0, 0, 0)).save(src)
    try:
        _jpeg_bytes_for_pikzels_frame(src)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "too small" in str(e).lower()
