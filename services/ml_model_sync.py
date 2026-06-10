"""
Download trained ML joblib artifacts from Hugging Face model repos on startup.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger("uploadm8.ml_model_sync")


def _repo_file_list(repo_id: str, filename: str) -> bool:
    try:
        from huggingface_hub import hf_hub_download

        token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or "").strip()
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            token=token or None,
        )
        return bool(path and Path(path).is_file())
    except Exception as e:
        logger.debug("hf download %s/%s failed: %s", repo_id, filename, e)
        return False


def sync_ml_models_from_hub() -> Dict[str, Any]:
    """Best-effort pull of promo + content model joblibs into data/ml/."""
    root = Path(__file__).resolve().parents[1] / "data" / "ml"
    root.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Any] = {"ok": True, "synced": []}

    from services.ml_hub_config import get_ml_hub_urls

    hub = get_ml_hub_urls()
    promo_repo = (os.environ.get("UM8_HF_MODEL_REPO") or hub.get("model_repo") or "").strip()
    if promo_repo:
        dest = root / "promo_uplift_model.joblib"
        if _repo_file_list(promo_repo, "promo_uplift_model.joblib"):
            try:
                from huggingface_hub import hf_hub_download

                token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or "").strip()
                p = hf_hub_download(
                    repo_id=promo_repo,
                    filename="promo_uplift_model.joblib",
                    local_dir=str(root),
                    token=token or None,
                )
                if p:
                    out["synced"].append({"model": "promo", "path": str(p)})
            except Exception as e:
                out["promo_error"] = str(e)[:200]

    content_repo = (os.environ.get("UM8_HF_CONTENT_MODEL_REPO") or hub.get("content_model_repo") or "").strip()
    if content_repo:
        if _repo_file_list(content_repo, "content_success_model.joblib"):
            try:
                from huggingface_hub import hf_hub_download

                token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or "").strip()
                p = hf_hub_download(
                    repo_id=content_repo,
                    filename="content_success_model.joblib",
                    local_dir=str(root),
                    token=token or None,
                )
                if p:
                    out["synced"].append({"model": "content", "path": str(p)})
            except Exception as e:
                out["content_error"] = str(e)[:200]

    return out
