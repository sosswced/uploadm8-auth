"""Dynamic generic hard-ban registry (admin overlay + learn path)."""

from __future__ import annotations

from services.generic_hard_ban import (
    apply_add,
    apply_learn_hits,
    apply_remove,
    apply_restore,
    builtin_ban_slugs,
    effective_ban_slugs,
    empty_overlay,
    is_hard_banned_slug,
    m8_hard_ban_prompt_block,
    normalize_overlay,
)


def test_builtin_seed_includes_nature_and_colors():
    bans = builtin_ban_slugs()
    assert "nature" in bans
    assert "modeoftransport" in bans
    assert "blue" in bans
    assert "horizon" in bans


def test_admin_remove_unbans_builtin_slug():
    o = empty_overlay()
    o = apply_remove(o, ["nature"], by="test")
    eff = effective_ban_slugs(o)
    assert "nature" not in eff
    assert "modeoftransport" in eff  # still banned


def test_admin_add_and_restore():
    o = empty_overlay()
    o = apply_add(o, ["weirdtaxonomy"], by="test")
    assert "weirdtaxonomy" in effective_ban_slugs(o)
    o = apply_remove(o, ["weirdtaxonomy"], by="test")
    assert "weirdtaxonomy" not in effective_ban_slugs(o)
    o = apply_restore(o, ["weirdtaxonomy"], by="test")
    # restored from removed but not re-added — only banned if still in added/learned/builtin
    assert "weirdtaxonomy" not in effective_ban_slugs(o)
    o = apply_add(o, ["weirdtaxonomy"], by="test")
    assert "weirdtaxonomy" in effective_ban_slugs(o)


def test_learn_auto_promotes_after_threshold():
    o = empty_overlay()
    o["auto_promote_after"] = 2
    o = apply_learn_hits(o, ["brandnewfiller"], source="test")
    assert "brandnewfiller" not in effective_ban_slugs(o)
    o = apply_learn_hits(o, ["brandnewfiller"], source="test")
    assert "brandnewfiller" in effective_ban_slugs(o)
    assert o["learned"]["brandnewfiller"]["status"] == "approved"


def test_removed_blocks_learn_promotion():
    o = empty_overlay()
    o = apply_remove(o, ["keepthis"], by="test")
    o["auto_promote_after"] = 1
    o = apply_learn_hits(o, ["keepthis"], source="test")
    assert "keepthis" not in effective_ban_slugs(o)


def test_is_hard_banned_respects_overlay_memory(monkeypatch):
    import core.state as state
    import services.generic_hard_ban as ghb

    ghb.invalidate_ban_cache()
    state.admin_settings_cache["generic_hard_ban"] = normalize_overlay(
        apply_add(empty_overlay(), ["custombanxyz"], by="test")
    )
    ghb.invalidate_ban_cache()
    assert is_hard_banned_slug("custombanxyz")
    assert is_hard_banned_slug("nature")
    # cleanup
    state.admin_settings_cache["generic_hard_ban"] = empty_overlay()
    ghb.invalidate_ban_cache()


def test_m8_prompt_block_mentions_registry():
    block = m8_hard_ban_prompt_block(limit=20)
    assert "HARD-BAN REGISTRY" in block
    assert "nature" in block or "NEVER use" in block


def test_normalize_overlay_caps_and_rejects_junk_chars():
    o = normalize_overlay(
        {
            "added": ["Nature!!", "blue", "", "x" * 80],
            "removed": ["mode of transport"],
            "learned": {"weird-tag": {"count": "3", "status": "pending"}},
            "auto_promote_after": 99,
        }
    )
    assert "nature" in o["added"]
    assert "blue" in o["added"]
    assert all(len(s) <= 48 for s in o["added"])
    assert "modeoftransport" in o["removed"]
    assert o["auto_promote_after"] == 50  # clamped
    assert o["learned"]["weirdtag"]["count"] == 3


def test_removed_wins_over_builtin_and_added():
    o = apply_add(empty_overlay(), ["nature", "customslug"], by="t")
    o = apply_remove(o, ["nature", "customslug"], by="t")
    eff = effective_ban_slugs(o)
    assert "nature" not in eff
    assert "customslug" not in eff
