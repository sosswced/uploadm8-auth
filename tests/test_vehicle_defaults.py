"""Garage default vehicle only stamps when Studio niche is automotive."""

from __future__ import annotations

from types import SimpleNamespace

from services.upload.vehicle_defaults import (
    resolve_presign_vehicle_ids,
    studio_niche_is_automotive,
)


def test_studio_niche_automotive_only():
    assert studio_niche_is_automotive(
        {"thumbnailStudioDefaultStrategy": {"audience_niche": "automotive"}}
    )
    assert studio_niche_is_automotive(
        {"thumbnail_studio_default_strategy": {"audienceNiche": "dashcam"}}
    )
    assert not studio_niche_is_automotive(
        {"thumbnailStudioDefaultStrategy": {"audience_niche": "sports"}}
    )
    assert not studio_niche_is_automotive({})


def test_presign_skips_default_mustang_when_not_automotive():
    data = SimpleNamespace(vehicle_make_id=None, vehicle_model_id=None)
    prefs = {
        "default_vehicle_make_id": 99,
        "default_vehicle_model_id": 7,
        "thumbnailStudioDefaultStrategy": {"audience_niche": "gardening"},
    }
    assert resolve_presign_vehicle_ids(data, prefs) == (None, None)


def test_presign_applies_default_when_automotive():
    data = SimpleNamespace(vehicle_make_id=None, vehicle_model_id=None)
    prefs = {
        "default_vehicle_make_id": 99,
        "default_vehicle_model_id": 7,
        "thumbnail_studio_default_strategy": {"audience_niche": "automotive"},
    }
    assert resolve_presign_vehicle_ids(data, prefs) == (99, 7)


def test_presign_explicit_vehicle_always_wins():
    data = SimpleNamespace(vehicle_make_id=12, vehicle_model_id=3)
    prefs = {
        "default_vehicle_make_id": 99,
        "thumbnailStudioDefaultStrategy": {"audience_niche": "sports"},
    }
    assert resolve_presign_vehicle_ids(data, prefs) == (12, 3)
