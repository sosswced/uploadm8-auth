"""Meta OAuth multi-destination pick helpers."""

from __future__ import annotations

from services.meta_oauth_connect import pick_destination


def test_pick_single_auto():
    dests = [{"destination_id": "1", "name": "Only"}]
    chosen, reason = pick_destination(dests)
    assert reason == "auto_single"
    assert chosen["destination_id"] == "1"


def test_pick_multi_needs_user():
    dests = [
        {"destination_id": "1", "name": "A"},
        {"destination_id": "2", "name": "B"},
    ]
    chosen, reason = pick_destination(dests)
    assert chosen is None
    assert reason == "need_pick"


def test_pick_reconnect_match():
    dests = [
        {"destination_id": "ig1", "page_id": "p1"},
        {"destination_id": "ig2", "page_id": "p2"},
    ]
    chosen, reason = pick_destination(dests, expected_provider_id="ig2")
    assert reason == "auto_reconnect"
    assert chosen["destination_id"] == "ig2"


def test_pick_empty():
    chosen, reason = pick_destination([])
    assert chosen is None
    assert reason == "none"
