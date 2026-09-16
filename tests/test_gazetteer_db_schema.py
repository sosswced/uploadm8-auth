"""Gazetteer lookup must match the loaded Census table, not only a geom schema."""

from services.gazetteer_db import gazetteer_lookup_plan, gazetteer_nearest_sql


def test_live_census_table_uses_text_centroids_not_missing_geom():
    cols = {
        "usps", "name", "intptlat", "intptlong", "geoid",
    }
    assert gazetteer_lookup_plan(cols) == "centroid"
    sql = gazetteer_nearest_sql(cols)
    assert "intptlat" in sql
    assert "intptlong" in sql
    assert "btrim(usps)" in sql
    assert "state_usps" not in sql
    assert "geom" not in sql


def test_postgis_geom_schema_still_uses_knn():
    cols = {"name", "state_usps", "geom"}
    assert gazetteer_lookup_plan(cols) == "geom"
    sql = gazetteer_nearest_sql(cols)
    assert "geom" in sql
    assert "state_usps" in sql


def test_empty_schema_is_not_a_lookup():
    assert gazetteer_lookup_plan(set()) == "none"
