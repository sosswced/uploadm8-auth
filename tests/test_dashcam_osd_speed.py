"""OSD speed accuracy — reject signage / OCR spikes before publish."""

from __future__ import annotations

from stages.dashcam_osd_stage import _aggregate, parse_osd_line


ESCORT_HUD = (
    "2025/03/05 04:50 12 PM 36.136162° -115.178398° 46MPH C Walker ESCORT."
)


def test_parse_escort_hud_speed():
    rec = parse_osd_line(ESCORT_HUD, t_s=1.0)
    assert rec["speed_mph"] == 46.0
    assert rec["speed_unit"] == "mph"
    assert rec["speed_hud_anchored"] is True
    assert rec["lat"] is not None


def test_reject_speed_limit_signage():
    rec = parse_osd_line("SPEED LIMIT 65 MPH", t_s=0.0)
    assert rec["speed_mph"] is None
    assert rec["speed_hud_anchored"] is False


def test_reject_speed_limit_embedded_with_hud_noise():
    # Signage context on the line must not yield a publishable speed.
    rec = parse_osd_line(
        "SPEED LIMIT 70 MPH ahead 36.136162° -115.178398°",
        t_s=0.0,
    )
    assert rec["speed_mph"] is None


def test_prefer_hud_speed_after_gps_over_earlier_sign_mph():
    line = "65 MPH 2025/03/05 04:50 12 PM 36.136162° -115.178398° 46MPH C Walker"
    rec = parse_osd_line(line, t_s=0.0)
    assert rec["speed_mph"] == 46.0
    assert rec["speed_hud_anchored"] is True


def test_vision_path_requires_hud_anchor_for_speed():
    bare = parse_osd_line("47 MPH", t_s=0.0, require_hud_anchor_for_speed=True)
    assert bare["speed_mph"] is None

    anchored = parse_osd_line(
        ESCORT_HUD, t_s=0.0, require_hud_anchor_for_speed=True
    )
    assert anchored["speed_mph"] == 46.0


def test_aggregate_drops_ocr_digit_spike():
    samples = [
        parse_osd_line(
            "2025/03/05 04:50 12 PM 36.136162° -115.178398° 46MPH C Walker",
            t_s=float(i),
        )
        for i in range(5)
    ]
    # Classic OCR bleed: lat leading digit glued onto speed → 146MPH once.
    spike = parse_osd_line(
        "2025/03/05 04:50 12 PM 36.136162° -115.178398° 146MPH C Walker",
        t_s=5.0,
    )
    samples.append(spike)
    osd = _aggregate(samples)
    assert osd["max_speed_mph"] == 46.0
    assert osd["speed_quality"]["speed_rejected_outlier"] >= 1


def test_aggregate_ignores_unanchored_sign_speeds_when_hud_present():
    hud = [
        parse_osd_line(ESCORT_HUD, t_s=float(i))
        for i in range(3)
    ]
    signs = [
        parse_osd_line("55 MPH", t_s=10.0),
        parse_osd_line("70 MPH", t_s=11.0),
    ]
    # Unanchored sign lines still parse speed without require_hud_anchor,
    # but aggregation must prefer HUD-anchored samples.
    assert signs[0]["speed_mph"] == 55.0
    assert signs[0]["speed_hud_anchored"] is False
    osd = _aggregate(hud + signs)
    assert osd["max_speed_mph"] == 46.0
    assert osd["speed_quality"]["speed_rejected_unanchored"] >= 1


def test_aggregate_rejects_lone_unanchored_speed():
    samples = [parse_osd_line("65 MPH", t_s=0.0)]
    osd = _aggregate(samples)
    assert osd["max_speed_mph"] == 0.0
    assert osd["speed_quality"]["speed_rejected_unconfirmed_peak"] >= 1


def test_aggregate_keeps_single_anchored_hud_speed():
    samples = [parse_osd_line(ESCORT_HUD, t_s=0.0)]
    osd = _aggregate(samples)
    assert osd["max_speed_mph"] == 46.0


def test_kph_converted_and_anchored():
    line = "2025/03/05 04:50:12 36.136162° -115.178398° 80 KPH"
    rec = parse_osd_line(line, t_s=0.0)
    assert rec["speed_unit"] == "kph"
    assert rec["speed_mph"] is not None
    assert 49.0 <= rec["speed_mph"] <= 50.5
    assert rec["speed_hud_anchored"] is True


def test_gps_path_does_not_carry_rejected_spike_speed():
    samples = [
        parse_osd_line(
            f"2025/03/05 04:50 12 PM 36.1361{i}° -115.1783{i}° 46MPH",
            t_s=float(i),
        )
        for i in range(4)
    ]
    samples.append(
        parse_osd_line(
            "2025/03/05 04:50 12 PM 36.136199° -115.178399° 146MPH",
            t_s=4.0,
        )
    )
    osd = _aggregate(samples)
    assert osd["max_speed_mph"] == 46.0
    for row in osd["gps_path"]:
        assert row[2] <= 50.0
