"""Unit tests for E2E media library random pair selection."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.e2e.helpers import media_library as ml
from tests.e2e.helpers.browser_session import SingleBrowserSession

ROOT = Path(__file__).resolve().parents[1]


def test_list_matching_pairs_same_stem(tmp_path: Path):
    (tmp_path / "clip_a.MP4").write_bytes(b"v")
    (tmp_path / "clip_a.map").write_text("m")
    (tmp_path / "orphan.MP4").write_bytes(b"v")
    (tmp_path / "clip_b.map").write_text("m")
    pairs = ml.list_matching_pairs(tmp_path)
    assert len(pairs) == 1
    assert pairs[0][0].stem == "clip_a"
    assert pairs[0][1].suffix.lower() == ".map"


def test_pick_random_respects_seed(tmp_path: Path, monkeypatch):
    for i in range(5):
        (tmp_path / f"c{i}.mp4").write_bytes(b"v")
        (tmp_path / f"c{i}.map").write_text("m")
    monkeypatch.setenv("E2E_MEDIA_PAIR_SEED", "42")
    ml.clear_media_pair_cache()
    a = ml.pick_random_media_pair(tmp_path, force_new=True)
    ml.clear_media_pair_cache()
    b = ml.pick_random_media_pair(tmp_path, force_new=True)
    assert a is not None and b is not None
    assert a[0].name == b[0].name


def test_cached_pair_stable(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("E2E_MEDIA_PAIR_SEED", raising=False)
    (tmp_path / "x.mp4").write_bytes(b"v")
    (tmp_path / "x.map").write_text("m")
    (tmp_path / "y.mp4").write_bytes(b"v")
    (tmp_path / "y.map").write_text("m")
    ml.clear_media_pair_cache()
    first = ml.pick_random_media_pair(tmp_path, force_new=True)
    second = ml.pick_random_media_pair(tmp_path, force_new=False)
    assert first == second


# --- resolve_demo_paths: never silently re-post the same fixed clip ---------


def test_journey_script_has_no_hardcoded_fixture_default():
    """Hard-coded --video default bypassed the library and re-posted the same
    clip live on every TUP run (root cause of duplicate posts on all platforms)."""
    src = (ROOT / "scripts" / "run_live_demo_journey.py").read_text(encoding="utf-8")
    assert "20250301_0058_CAM_EVNT" not in src
    assert "20250224_0073_CAM_EVNT" not in src


def test_resolve_demo_paths_uses_library_pick(tmp_path: Path, monkeypatch):
    (tmp_path / "fresh.mp4").write_bytes(b"v")
    (tmp_path / "fresh.map").write_text("m")
    monkeypatch.setenv("E2E_MEDIA_LIBRARY", str(tmp_path))
    monkeypatch.delenv("E2E_TEST_VIDEO", raising=False)
    monkeypatch.delenv("E2E_TEST_TELEMETRY_MAP", raising=False)
    monkeypatch.delenv("E2E_MEDIA_PAIR_SEED", raising=False)
    ml.clear_media_pair_cache()
    from tests.e2e.helpers.live_demo import resolve_demo_paths

    v, t = resolve_demo_paths()
    assert v.name == "fresh.mp4"
    assert t is not None and t.name == "fresh.map"


def test_resolve_demo_paths_raises_when_library_configured_but_empty(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("E2E_MEDIA_LIBRARY", str(tmp_path))
    monkeypatch.delenv("E2E_TEST_VIDEO", raising=False)
    monkeypatch.delenv("E2E_TEST_TELEMETRY_MAP", raising=False)
    ml.clear_media_pair_cache()
    from tests.e2e.helpers.live_demo import resolve_demo_paths

    with pytest.raises(FileNotFoundError, match="media library"):
        resolve_demo_paths()


def test_resolve_demo_paths_explicit_video_wins(tmp_path: Path, monkeypatch):
    explicit = tmp_path / "explicit.mp4"
    explicit.write_bytes(b"v")
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "other.mp4").write_bytes(b"v")
    (lib / "other.map").write_text("m")
    monkeypatch.setenv("E2E_MEDIA_LIBRARY", str(lib))
    monkeypatch.setenv("E2E_TEST_VIDEO", str(explicit))
    monkeypatch.delenv("E2E_TEST_TELEMETRY_MAP", raising=False)
    ml.clear_media_pair_cache()
    from tests.e2e.helpers.live_demo import resolve_demo_paths

    v, _ = resolve_demo_paths()
    assert v == explicit


def test_human_guards_accept_duplicate_upload_confirm():
    """Playwright must accept the allowDuplicate confirm or TUP aborts as Skipped."""
    import inspect

    src = inspect.getsource(SingleBrowserSession._attach_human_guards)
    assert "duplicate" in src
    assert "dialog.accept()" in src
    assert 'kind == "confirm"' in src or 'type == "confirm"' in src or '== "confirm"' in src


def test_tup_mode_refuses_fixed_fixture_fallback(tmp_path: Path, monkeypatch):
    """Under E2E_TUP=1, missing library/explicit must not return DEFAULT_E2E_VIDEO."""
    monkeypatch.setenv("E2E_TUP", "1")
    monkeypatch.delenv("E2E_TEST_VIDEO", raising=False)
    monkeypatch.delenv("E2E_TEST_TELEMETRY_MAP", raising=False)
    # Point library at empty dir so pick returns None.
    monkeypatch.setenv("E2E_MEDIA_LIBRARY", str(tmp_path))
    ml.clear_media_pair_cache()
    from tests.e2e.helpers import config as cfg

    assert cfg.e2e_test_video() is None
    assert cfg.e2e_test_telemetry_map() is None


def test_library_pair_skipped_when_only_one_explicit_env(tmp_path: Path, monkeypatch):
    """Either explicit env must disable the library — otherwise video and .map diverge."""
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "libclip.mp4").write_bytes(b"v")
    (lib / "libclip.map").write_text("m")
    explicit = tmp_path / "only_video.mp4"
    explicit.write_bytes(b"v")
    monkeypatch.setenv("E2E_MEDIA_LIBRARY", str(lib))
    monkeypatch.setenv("E2E_TEST_VIDEO", str(explicit))
    monkeypatch.delenv("E2E_TEST_TELEMETRY_MAP", raising=False)
    ml.clear_media_pair_cache()
    from tests.e2e.helpers import config as cfg

    assert cfg._library_pair() is None
    assert cfg.e2e_test_video() == explicit
    # Must NOT pull a random library map when only video is explicit.
    assert cfg.e2e_test_telemetry_map() != (lib / "libclip.map")
