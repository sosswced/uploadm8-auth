"""Speed units: MPH/KMH only — never degrees or lat/lon."""

from __future__ import annotations

from core.speed_units import is_speed_unit, looks_like_coordinate_or_degree, normalize_speed_unit
from services.hydration_enforcer import _vision_ocr_peak_mph
from stages.dashcam_osd_stage import parse_osd_line


def test_is_speed_unit_accepts_mph_kmh_only():
    assert is_speed_unit("mph")
    assert is_speed_unit("MPH")
    assert is_speed_unit("km/h")
    assert is_speed_unit("KPH")
    assert not is_speed_unit("°")
    assert not is_speed_unit("deg")
    assert not is_speed_unit("HDG")
    assert not is_speed_unit("")
    assert normalize_speed_unit("km/h") == "kph"
    assert normalize_speed_unit("mi/h") == "mph"


def test_looks_like_coordinate_or_degree():
    assert looks_like_coordinate_or_degree("36.136162° -115.178398°")
    assert looks_like_coordinate_or_degree("270° HDG")
    assert looks_like_coordinate_or_degree("32 degrees")
    assert not looks_like_coordinate_or_degree(
        "36.136162° -115.178398° 88 MPH C Walker"
    )


def test_parse_osd_ignores_latlon_and_heading_without_unit():
    bare = parse_osd_line("36.136162° -115.178398°")
    assert bare.get("speed_mph") is None
    hdg = parse_osd_line("270° HDG 36.136162° -115.178398°")
    assert hdg.get("speed_mph") is None
    ok = parse_osd_line("2025/03/05 04:50 12 PM 36.136162° -115.178398° 88MPH C Walker")
    assert ok.get("speed_mph") == 88.0
    assert ok.get("speed_unit") == "mph"


def test_reject_lon_integer_glued_to_mph():
    """Classic OCR bleed: western lon (-115 / -122) glued onto a unit."""
    from core.speed_units import speed_match_is_coordinate_bleed

    assert speed_match_is_coordinate_bleed(
        "-115MPH", match_start=1, match_end=7, speed_digits="115"
    )
    for line in (
        "2025/03/05 04:50 12 PM 36.136162° -115MPH C Walker",
        "36.136162° -122MPH",
        "41.92226° -122.57585°MPH",  # unit glued to lon decimal → fraction/deg bleed
        "36.136162° -115.178398°115°MPH",  # lon integer + degree + unit
        "36.136162° -115.178398° 115°MPH",  # degree-bound lon int, not HUD speed
    ):
        rec = parse_osd_line(line, t_s=0.0)
        assert rec.get("speed_mph") is None, line


def test_reject_lon_fraction_tail_as_speed():
    """``115.178398°MPH`` must not become 398 MPH (or any fraction chunk)."""
    rec = parse_osd_line(
        "2025/03/05 04:50 12 PM 36.136162° -115.178398°MPH C Walker",
        t_s=0.0,
    )
    assert rec.get("speed_mph") is None


def test_real_hud_speed_still_reads_after_full_gps():
    """Separate HUD sample after a complete lat/lon pair is still valid —
    including when the speed happens to equal the lon integer (115)."""
    rec = parse_osd_line(
        "2025/03/05 04:50 12 PM 36.136162° -115.178398° 115MPH C Walker",
        t_s=0.0,
    )
    assert rec.get("speed_mph") == 115.0
    assert rec.get("speed_unit") == "mph"


def test_vision_ocr_peak_ignores_coordinates_and_headings():
    assert _vision_ocr_peak_mph("36.136162° -115.178398°\n270° HDG") == 0.0
    assert _vision_ocr_peak_mph("36.136162° -115MPH\n41.9° -122°MPH") == 0.0
    ocr = (
        "2025/03/05 04:50 12 PM 36.136162° -115.178398° 88MPH C Walker\n"
        "2025/03/05 04:51 12 PM 36.136200° -115.178400° 90MPH C Walker"
    )
    assert _vision_ocr_peak_mph(ocr) >= 88.0
