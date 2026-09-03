"""Escort .map speed column is km/h — must convert before publish/consensus."""

from __future__ import annotations

from pathlib import Path

from stages.telemetry_stage import _parse_nmea_point_from_row, parse_map_file


def test_escort_nmea_row_converts_kmh_to_mph():
    # Peak row from 20250226_0030_CAM.map at 16:56:51 — HUD shows 79MPH.
    row = "A,260225,165651,3608.8071,N,11510.1201,W,128.49,-0.03,0.26,1.75;".rstrip(
        ";"
    ).split(",")
    pt = _parse_nmea_point_from_row(row, fallback_ts=0.0)
    assert pt is not None
    assert abs(pt["speed_mph"] - 79.84) < 0.1


def test_parse_map_file_peak_matches_hud_not_raw_kmh(tmp_path: Path):
    # Minimal Escort snippet: raw 128.49 must not publish as 128 MPH.
    content = (
        "A,260225,165650,3608.7915,N,11510.1328,W,127.95,0.0,0.0,1.5;\r\n"
        "A,260225,165651,3608.8071,N,11510.1201,W,128.49,0.0,0.0,1.5;\r\n"
    )
    p = tmp_path / "sample.map"
    p.write_text(content, encoding="utf-8")
    tel = parse_map_file(p)
    assert tel.max_speed_mph < 90.0
    assert abs(tel.max_speed_mph - 79.84) < 0.2
