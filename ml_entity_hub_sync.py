"""
Sync visual entity catalogs to Hugging Face Datasets for ML training and evaluation.

Wires UploadM8's per-user entity memory into the Hub stack documented in
``services/ml_hub_config`` (TRL Jobs, Trackio, hf CLI, evaluation model cards).

Set ``UM8_HF_DATASET_REPO`` (or ``UM8_HF_DATASET_URL``) and ``HF_TOKEN`` before calling.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from services.ml_hub_config import get_ml_hub_urls
from services.ml_observability import hf_write_token
from services.visual_entity_memory import catalog_rows_for_hf_export, fetch_channel_catalog_detail

logger = logging.getLogger("uploadm8.ml_entity_hub_sync")

VISUAL_ENTITIES_CONFIG = "visual_entities"


async def build_platform_visual_entity_export(
    db_pool,
    *,
    limit_per_user: int = 200,
) -> List[Dict[str, Any]]:
    """Export recent per-user catalog rows for Hub dataset append."""
    if not db_pool:
        return []
    hub = get_ml_hub_urls()
    repo = hub.get("dataset_repo") or ""
    rows: List[Dict[str, Any]] = []
    try:
        async with db_pool.acquire() as conn:
            users = await conn.fetch(
                """
                SELECT user_id::text AS user_id,
                       MAX(last_category) AS category
                  FROM user_visual_entity_catalog
                 GROUP BY user_id
                 ORDER BY MAX(last_seen_at) DESC NULLS LAST
                 LIMIT $1
                """,
                limit_per_user,
            )
        for u in users:
            uid = str(u["user_id"])
            cat = str(u.get("category") or "general")
            catalog = await fetch_channel_catalog_detail(
                db_pool, user_id=uid, category=cat, limit_per_bucket=16
            )
            rows.extend(
                catalog_rows_for_hf_export(catalog, user_id=uid, hub_dataset_repo=repo)
            )
    except Exception as e:
        logger.warning("[ml_entity_hub_sync] export build failed: %s", e)
    return rows


def push_visual_entities_to_hub(
    rows: List[Dict[str, Any]],
    *,
    dataset_repo: Optional[str] = None,
    config_name: str = VISUAL_ENTITIES_CONFIG,
) -> Dict[str, Any]:
    """
    Push rows to a Hub dataset config (creates config if missing).
    Returns status dict; does not raise on missing token (caller handles UI).
    """
    token = hf_write_token()
    repo = (dataset_repo or "").strip() or (get_ml_hub_urls().get("dataset_repo") or "")
    if not token:
        return {"ok": False, "error": "HF_TOKEN not configured"}
    if not repo:
        return {"ok": False, "error": "UM8_HF_DATASET_REPO not configured"}
    if not rows:
        return {"ok": True, "rows": 0, "message": "nothing to push"}

    try:
        from datasets import Dataset
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.create_repo(repo_id=repo, repo_type="dataset", exist_ok=True)
        ds = Dataset.from_list(rows)
        ds.push_to_hub(
            repo,
            config_name=config_name,
            token=token,
            commit_message=f"UploadM8 visual entities {datetime.now(timezone.utc).isoformat()}",
        )
        return {
            "ok": True,
            "rows": len(rows),
            "dataset_repo": repo,
            "config": config_name,
            "dataset_url": f"https://huggingface.co/datasets/{repo}",
        }
    except Exception as e:
        logger.warning("[ml_entity_hub_sync] hub push failed: %s", e)
        return {"ok": False, "error": str(e)[:400]}


async def sync_visual_entities_for_user(
    db_pool,
    *,
    user_id: str,
    category: str = "general",
) -> Dict[str, Any]:
    """Export one user's catalog and append to Hub (best-effort)."""
    if os.environ.get("UM8_HF_SYNC_VISUAL_ENTITIES", "").strip().lower() not in (
        "1",
        "true",
        "yes",
    ):
        return {"ok": False, "skipped": True, "reason": "UM8_HF_SYNC_VISUAL_ENTITIES disabled"}
    catalog = await fetch_channel_catalog_detail(
        db_pool, user_id=user_id, category=category, limit_per_bucket=24
    )
    hub = get_ml_hub_urls()
    rows = catalog_rows_for_hf_export(
        catalog, user_id=user_id, hub_dataset_repo=hub.get("dataset_repo") or ""
    )
    return push_visual_entities_to_hub(rows)
