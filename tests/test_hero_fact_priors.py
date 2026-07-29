"""Phase 5 hero-fact class priors."""

from __future__ import annotations

from core.hero_fact_priors import (
    class_rank_for_cluster,
    rank_hero_facts,
    rebuild_hero_fact_priors,
)


def test_bootstrap_rank_puts_landmark_before_speed():
    from core.hero_fact_priors import _BOOTSTRAP_GLOBAL

    # Force bootstrap priors — local data/ml/hero_fact_priors_v1.json may be learned.
    order = class_rank_for_cluster(
        "gardening",
        priors={"version": 1, "global": list(_BOOTSTRAP_GLOBAL), "clusters": {}},
    )
    assert order.index("landmark") < order.index("speed")


def test_rank_hero_facts_reorders_by_cluster_prior(tmp_path):
    priors = {
        "version": 1,
        "global": ["entity", "speed"],
        "clusters": {"gardening": ["count", "entity", "speed"]},
        "source": "test",
    }
    facts = [
        {"text": "fast drive", "class": "speed", "score": 9.0},
        {"text": "40 tomatoes", "class": "count", "score": 2.0},
        {"text": "raised bed", "class": "entity", "score": 5.0},
    ]
    ranked = rank_hero_facts(facts, domain_tag="gardening", priors=priors)
    assert [f["class"] for f in ranked] == ["count", "entity", "speed"]


def test_rebuild_writes_cluster_when_enough_rows(tmp_path):
    rows = []
    for i in range(30):
        rows.append({
            "identity_domain_tag": "gardening",
            "identity_headline_class": "count" if i % 2 == 0 else "entity",
            "is_hot": 1 if i % 2 == 0 else 0,
            "hotness_score": 3.0 if i % 2 == 0 else 0.5,
        })
    out = tmp_path / "priors.json"
    payload = rebuild_hero_fact_priors(rows, out_path=out, min_rows=25)
    assert out.exists()
    assert payload["source"] == "learned"
    assert "gardening" in payload["clusters"]
    assert payload["clusters"]["gardening"][0] == "count"


def test_rebuild_below_min_rows_keeps_global_only(tmp_path):
    rows = [
        {
            "identity_domain_tag": "food",
            "identity_headline_class": "entity",
            "is_hot": 1,
            "hotness_score": 2.0,
        }
    ] * 5
    out = tmp_path / "priors.json"
    payload = rebuild_hero_fact_priors(rows, out_path=out, min_rows=25)
    assert payload["clusters"] == {}
    assert "entity" in payload["global"]
