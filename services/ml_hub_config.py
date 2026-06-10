"""
Single source of truth for Hugging Face / Trackio URLs exposed to the web app.

Used by ``routers/features`` (public ml-hub) and ``routers/admin`` (observability)
so env overrides never drift between surfaces.

**Important:** There are no baked-in third-party Hub URLs. If ``UM8_HF_DATASET_REPO`` /
``UM8_TRACKIO_SPACE_PATH`` (or explicit ``*_URL`` overrides) are unset, the API returns
``null`` URLs and the UI links to official docs instead of a 404.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Dict, List, Optional, Tuple

HUB_DOCS_JOBS = "https://huggingface.co/docs/huggingface_hub/guides/jobs"
DATASETS_HUB_DOC = "https://huggingface.co/docs/datasets"
DATASET_CREATE_DOC = "https://huggingface.co/docs/datasets/upload_dataset"
SPACES_OVERVIEW_DOC = "https://huggingface.co/docs/hub/spaces-overview"
TRACKIO_DOC = "https://huggingface.co/docs/trackio/en/quickstart"
TRL_ROOT_DOC = "https://huggingface.co/docs/trl"
TRL_JOBS_DOC = "https://huggingface.co/docs/trl/en/jobs_training"
MODEL_CARDS_DOC = "https://huggingface.co/docs/hub/model-cards"
EVALUATION_DOC = "https://huggingface.co/docs/hub/eval-results"


def _env(name: str, default: str = "") -> str:
    v = (os.environ.get(name) or "").strip()
    return v if v else default


def _parse_hub_page_response(status_code: int, body: str) -> bool:
    if status_code in (401, 403):
        return True
    text = (body or "")[:12000].lower()
    if "can't find the page" in text:
        return False
    if "<title>" in text:
        title = text.split("<title>", 1)[1].split("</title>", 1)[0]
        if title.strip().startswith("404") and "hugging face" in title:
            return False
    if status_code >= 400:
        return False
    return True


def hub_public_page_exists(url: str) -> bool:
    """
    Best-effort check that a Hub dataset/space URL is not a generic 404 page.

    Private repos may return 401/403 — treated as reachable. Network errors are
    treated as reachable so we do not hide configured links offline.
    """
    u = (url or "").strip()
    if not u.startswith("https://huggingface.co/"):
        return False
    if os.environ.get("UM8_ML_OBSERVABILITY_SKIP_HF_PROBE", "").strip() == "1":
        return True
    try:
        import httpx

        r = httpx.get(
            u,
            follow_redirects=True,
            timeout=8.0,
            headers={"User-Agent": "UploadM8-MLHubCheck/1.0"},
        )
        return _parse_hub_page_response(r.status_code, r.text or "")
    except Exception:
        return True


async def hub_public_page_exists_async(url: str) -> bool:
    """Non-blocking Hub page probe for async routes."""
    u = (url or "").strip()
    if not u.startswith("https://huggingface.co/"):
        return False
    if os.environ.get("UM8_ML_OBSERVABILITY_SKIP_HF_PROBE", "").strip() == "1":
        return True
    try:
        import httpx

        async with httpx.AsyncClient(follow_redirects=True, timeout=8.0) as client:
            r = await client.get(u, headers={"User-Agent": "UploadM8-MLHubCheck/1.0"})
        return _parse_hub_page_response(r.status_code, r.text or "")
    except Exception:
        return True


async def check_hub_pages_parallel(dataset_url: str, space_url: str) -> Tuple[bool, bool]:
    """Probe dataset + space URLs concurrently (max ~8s, not 16s sequential)."""
    ds, sp = await asyncio.gather(
        hub_public_page_exists_async(dataset_url or ""),
        hub_public_page_exists_async(space_url or ""),
    )
    return bool(ds), bool(sp)


_OBSERVABILITY_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}
OBSERVABILITY_CACHE_TTL_SEC = int(os.environ.get("UM8_ML_OBSERVABILITY_CACHE_SEC", "600") or "600")


def get_observability_cache(mode: str) -> Optional[Dict[str, Any]]:
    key = f"observability:{mode}"
    entry = _OBSERVABILITY_CACHE.get(key)
    if not entry:
        return None
    expires_at, payload = entry
    if time.time() >= expires_at:
        _OBSERVABILITY_CACHE.pop(key, None)
        return None
    return payload


def set_observability_cache(mode: str, payload: Dict[str, Any]) -> None:
    key = f"observability:{mode}"
    _OBSERVABILITY_CACHE[key] = (time.time() + OBSERVABILITY_CACHE_TTL_SEC, payload)


def clear_observability_cache() -> None:
    _OBSERVABILITY_CACHE.clear()


def get_ml_hub_urls() -> Dict[str, Optional[str]]:
    """
    Resolve Hub URLs from environment only (no default namespace/repo).

    Set ``UM8_HF_DATASET_REPO`` (e.g. ``your-org/promo-targeting-v1``) and/or
    ``UM8_TRACKIO_SPACE_PATH`` (e.g. ``your-org/uploadm8-trackio``), or set full
    ``UM8_HF_DATASET_URL`` / ``UM8_TRACKIO_SPACE_URL`` explicitly.
    """
    dataset_repo = _env("UM8_HF_DATASET_REPO")
    dataset_url = _env("UM8_HF_DATASET_URL")
    if not dataset_url and dataset_repo:
        dataset_url = f"https://huggingface.co/datasets/{dataset_repo}"

    trackio_space_path = _env("UM8_TRACKIO_SPACE_PATH")
    trackio_space_url = _env("UM8_TRACKIO_SPACE_URL")
    if not trackio_space_url and trackio_space_path:
        trackio_space_url = f"https://huggingface.co/spaces/{trackio_space_path}"

    # Content-success loop owns its own dataset + model "bucket" (separate schema).
    content_dataset_repo = _env("UM8_HF_CONTENT_DATASET_REPO")
    content_model_repo = _env("UM8_HF_CONTENT_MODEL_REPO")
    promo_owner = ""
    for repo in (dataset_repo, _env("UM8_HF_MODEL_REPO")):
        if repo and "/" in repo:
            promo_owner = repo.split("/", 1)[0]
            break
    if promo_owner:
        if not content_dataset_repo:
            content_dataset_repo = f"{promo_owner}/uploadm8-content-success-v1"
        if not content_model_repo:
            content_model_repo = f"{promo_owner}/uploadm8-content-success-model-v1"

    content_dataset_url = _env("UM8_HF_CONTENT_DATASET_URL")
    if not content_dataset_url and content_dataset_repo:
        content_dataset_url = f"https://huggingface.co/datasets/{content_dataset_repo}"
    content_model_url = _env("UM8_HF_CONTENT_MODEL_URL")
    if not content_model_url and content_model_repo:
        content_model_url = f"https://huggingface.co/{content_model_repo}"

    return {
        "dataset_repo": dataset_repo or None,
        "trackio_space_path": trackio_space_path or None,
        "dataset_url": dataset_url or None,
        "trackio_space_url": trackio_space_url or None,
        "model_repo": (_env("UM8_HF_MODEL_REPO") or None),
        "model_url": (
            _env("UM8_HF_MODEL_URL")
            or (
                f"https://huggingface.co/{_env('UM8_HF_MODEL_REPO')}"
                if _env("UM8_HF_MODEL_REPO")
                else None
            )
        ),
        "content_dataset_repo": content_dataset_repo or None,
        "content_dataset_url": content_dataset_url or None,
        "content_model_repo": content_model_repo or None,
        "content_model_url": content_model_url or None,
    }


def ml_hub_huggingface_dict() -> Dict[str, Any]:
    """Public Hub links (no secrets). URL values may be null until env is configured."""
    u = get_ml_hub_urls()
    return {
        "dataset_repo": u["dataset_repo"],
        "dataset_url": u["dataset_url"],
        "trackio_space_url": u["trackio_space_url"],
        "trackio_space_path": u["trackio_space_path"],
        "model_repo": u.get("model_repo"),
        "model_url": u.get("model_url"),
        "content_dataset_repo": u.get("content_dataset_repo"),
        "content_dataset_url": u.get("content_dataset_url"),
        "content_model_repo": u.get("content_model_repo"),
        "content_model_url": u.get("content_model_url"),
        "hub_docs_jobs": HUB_DOCS_JOBS,
        "datasets_hub": DATASETS_HUB_DOC,
        "trl_docs": TRL_ROOT_DOC,
        "model_cards_doc": MODEL_CARDS_DOC,
        "evaluation_doc": EVALUATION_DOC,
        "trackio_doc": TRACKIO_DOC,
        "setup_guide_anchor": "guide.html#feat-hf-assets-setup",
        "dataset_help_url": DATASET_CREATE_DOC,
        "trackio_help_url": TRACKIO_DOC,
    }


def _resolved_hub_link(
    configured_url: str,
    *,
    fallback_doc: str,
    in_app_setup: str = "guide.html#feat-hf-assets-setup",
) -> tuple[str, bool]:
    """Return (url, placeholder). Unset or 404 Hub pages link to in-app setup (not empty Hub)."""
    u = (configured_url or "").strip()
    if not u:
        return in_app_setup, True
    if not hub_public_page_exists(u):
        return in_app_setup, True
    return u, False


def _ecosystem_entries(urls: Dict[str, Optional[str]]) -> List[Dict[str, Any]]:
    d_u = (urls.get("dataset_url") or "").strip()
    t_u = (urls.get("trackio_space_url") or "").strip()
    trackio_url, trackio_ph = _resolved_hub_link(t_u, fallback_doc=TRACKIO_DOC)
    dataset_url, dataset_ph = _resolved_hub_link(d_u, fallback_doc=DATASET_CREATE_DOC)
    return [
        {
            "id": "trackio",
            "label": "Trackio",
            "role": "Experiment metrics and training alerts synced to a Space dashboard.",
            "url": trackio_url,
            "placeholder": trackio_ph,
        },
        {
            "id": "datasets",
            "label": "Promo targeting dataset",
            "role": "Versioned promo-targeting uplift training table (split: train).",
            "url": dataset_url,
            "placeholder": dataset_ph,
        },
        {
            "id": "content_datasets",
            "label": "Content success dataset",
            "role": "Separate bucket for content-success / hottest-topic rows (split: train).",
            "url": _resolved_hub_link(
                (urls.get("content_dataset_url") or "").strip(),
                fallback_doc=DATASET_CREATE_DOC,
            )[0],
            "placeholder": _resolved_hub_link(
                (urls.get("content_dataset_url") or "").strip(),
                fallback_doc=DATASET_CREATE_DOC,
            )[1],
        },
        {
            "id": "evaluation",
            "label": "Model cards & evaluation",
            "role": "Structured benchmarks on the Hub model index.",
            "url": EVALUATION_DOC,
            "placeholder": False,
        },
        {
            "id": "trainer",
            "label": "TRL / Jobs training",
            "role": "Fine-tune with TRL on Hugging Face Jobs (GPU) with Hub push.",
            "url": TRL_JOBS_DOC,
            "placeholder": False,
        },
        {
            "id": "jobs",
            "label": "HF Jobs",
            "role": "Batch inference, data jobs, and scheduled UV workloads.",
            "url": HUB_DOCS_JOBS,
            "placeholder": False,
        },
    ]


def build_ml_hub_public_response() -> Dict[str, Any]:
    """Payload for ``GET /api/features/ml-hub``."""
    urls_raw = get_ml_hub_urls()
    urls_for_eco: Dict[str, Optional[str]] = {
        "dataset_url": urls_raw.get("dataset_url"),
        "content_dataset_url": urls_raw.get("content_dataset_url"),
        "trackio_space_url": urls_raw.get("trackio_space_url"),
    }
    return {
        "guide_anchor": "guide.html#feat-ml-ai-hub",
        "setup_guide_anchor": "guide.html#feat-hf-assets-setup",
        "configured": {
            "hf_dataset_page": bool(urls_raw.get("dataset_url")),
            "hf_content_dataset_page": bool(urls_raw.get("content_dataset_url")),
            "hf_trackio_space": bool(urls_raw.get("trackio_space_url")),
        },
        "huggingface": ml_hub_huggingface_dict(),
        "ecosystem": _ecosystem_entries(urls_for_eco),
    }
