"""Unit tests for ML strategy scoring helpers."""
import pytest

from services.ml_strategy_utils import empirical_bayes_mean, prefer_ai_thumbnail_vs_sharpness


def test_empirical_bayes_shrinks_toward_prior():
    # No data → prior
    assert empirical_bayes_mean(10.0, 0) == pytest.approx(2.0, rel=1e-3)
    # Lots of data → raw mean
    assert empirical_bayes_mean(10.0, 10_000) == pytest.approx(10.0, rel=1e-2)


def test_prefer_ai_requires_min_samples():
    use, d = prefer_ai_thumbnail_vs_sharpness(
        sharp_mean=1.0,
        sharp_samples=50,
        ai_mean=8.0,
        ai_samples=2,
        min_samples_ai=4,
    )
    assert use is False
    assert d["reason"] == "insufficient_ai_samples"


def test_prefer_ai_when_clearly_ahead():
    use, d = prefer_ai_thumbnail_vs_sharpness(
        sharp_mean=1.5,
        sharp_samples=40,
        ai_mean=6.0,
        ai_samples=30,
        min_samples_ai=4,
        margin=0.1,
    )
    assert use is True
    assert d["eb_ai"] > d["eb_sharp"]
