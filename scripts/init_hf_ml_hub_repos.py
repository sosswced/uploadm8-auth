# /// script
# dependencies = [
#   "huggingface_hub>=0.26.0",
#   "python-dotenv>=1.0.0,<2.0.0",
# ]
# ///
"""
Create Hugging Face dataset + Space repos for UploadM8 ML hub (idempotent).

Requires a write token: set ``HF_TOKEN`` (preferred) or ``HUGGING_FACE_HUB_TOKEN``.
After creation, set in .env::

    UM8_HF_DATASET_REPO=your-org/repo-name
    UM8_TRACKIO_SPACE_PATH=your-org/space-name

Or set full URLs with UM8_HF_DATASET_URL / UM8_TRACKIO_SPACE_URL.

Run::

    uv run scripts/init_hf_ml_hub_repos.py --dataset-repo YOUR/promo-targeting-v1 --trackio-space YOUR/uploadm8-trackio

Omit flags to use UM8_HF_DATASET_REPO and UM8_TRACKIO_SPACE_PATH from the environment.

Re-running after an upgrade updates the Space ``README.md`` / ``app.py`` / ``requirements.txt``
so a Space that shows "Missing configuration in README" can be repaired without deleting the repo.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _die(msg: str, code: int = 1) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(code)


def _hf_write_token() -> str:
    return (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()


# Hugging Face Spaces require YAML frontmatter in README.md plus the declared app_file.
# See https://huggingface.co/docs/hub/spaces-config-reference
_SPACE_README = """---
title: UploadM8 Trackio
emoji: 📈
colorFrom: gray
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
license: mit
tags:
 - trackio
datasets:
 - cedy243/uploadm8-promo-targeting-v1
---

# UploadM8 + Trackio

Live experiment dashboard for **uploadm8-ml** runs logged from UploadM8 training jobs
(`TRACKIO_PROJECT` / `TRACKIO_SPACE_ID` in the API `.env`).
"""

# Trackio ships its own Starlette dashboard (trackio.show). Do not pin legacy Gradio 4.x here —
# HF still installs Gradio for sdk:gradio Spaces, but app.py uses Trackio only.
_SPACE_REQUIREMENTS = """trackio[spaces,mcp]>=0.25.0,<1.0.0
pyarrow>=21.0
"""

_SPACE_APP = '''"""UploadM8 Trackio experiment dashboard on Hugging Face Spaces."""
import os

import trackio
from trackio.sqlite_storage import SQLiteStorage

# Space containers are ephemeral — restore SQLite from the HF Bucket before serving.
SQLiteStorage.load_from_dataset()

_project = (os.environ.get("TRACKIO_PROJECT") or "uploadm8-ml").strip()
trackio.show(project=_project or None, open_browser=False)
'''

# Writable inside HF Docker without a paid persistent volume. Metrics persist in TRACKIO_BUCKET_ID.
_TRACKIO_DIR_ON_SPACE = "/app/trackio"


def _trackio_bucket_id(space_id: str) -> str:
    return f"{space_id}-bucket"


def _configure_trackio_space(api, space_id: str, *, token: str, project: str) -> None:
    """Wire bucket vars, secrets, and optional /data volume for Trackio on Spaces."""
    bucket_id = _trackio_bucket_id(space_id)
    try:
        import trackio
        from trackio.bucket_storage import create_bucket_if_not_exists
        from trackio.deploy import _ensure_bucket_mounted_at_data

        create_bucket_if_not_exists(bucket_id, private=None)
        try:
            _ensure_bucket_mounted_at_data(space_id, bucket_id, hf_api=api)
        except Exception as exc:
            print(f"[warn] bucket volume attach skipped: {exc}")
    except ImportError:
        print("[warn] trackio not installed locally; skipping bucket create/mount")

    import huggingface_hub

    for key, value in (
        ("TRACKIO_DIR", _TRACKIO_DIR_ON_SPACE),
        ("TRACKIO_BUCKET_ID", bucket_id),
        ("TRACKIO_PROJECT", project),
    ):
        huggingface_hub.add_space_variable(space_id, key, value)
    huggingface_hub.add_space_secret(space_id, "HF_TOKEN", token)
    print(f"[ok] space vars: TRACKIO_DIR={_TRACKIO_DIR_ON_SPACE}, bucket={bucket_id}, project={project}")


def main() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(_REPO_ROOT / ".env")
    except ImportError:
        pass

    p = argparse.ArgumentParser(description="Create HF dataset + Trackio Space repos (empty, idempotent).")
    p.add_argument(
        "--dataset-repo",
        default=(os.environ.get("UM8_HF_DATASET_REPO") or "").strip(),
        help="Hub dataset repo id (e.g. org/promo-targeting-v1). Default: UM8_HF_DATASET_REPO",
    )
    p.add_argument(
        "--trackio-space",
        default=(os.environ.get("UM8_TRACKIO_SPACE_PATH") or "").strip(),
        help="Hub Space repo id for Trackio (e.g. org/uploadm8-trackio). Default: UM8_TRACKIO_SPACE_PATH",
    )
    p.add_argument(
        "--model-repo",
        default=(os.environ.get("UM8_HF_MODEL_REPO") or "").strip(),
        help="Hub model repo for eval results (e.g. org/uploadm8-promo-uplift). Default: UM8_HF_MODEL_REPO",
    )
    p.add_argument("--private", action="store_true", help="Create private repos.")
    p.add_argument(
        "--sync-trackio",
        action="store_true",
        help="After Space upload, sync local uploadm8-ml metrics to the Space bucket.",
    )
    args = p.parse_args()

    token = _hf_write_token()
    if not token:
        _die("HF_TOKEN or HUGGING_FACE_HUB_TOKEN is required (write token).")

    if not args.dataset_repo and not args.trackio_space and not args.model_repo:
        _die(
            "Provide --dataset-repo, --trackio-space, and/or --model-repo, or set "
            "UM8_HF_DATASET_REPO / UM8_TRACKIO_SPACE_PATH / UM8_HF_MODEL_REPO in the environment."
        )

    from huggingface_hub import HfApi

    from services.ml_eval_hub import ensure_model_repo, push_dataset_eval_yaml

    api = HfApi(token=token)

    if args.dataset_repo:
        api.create_repo(args.dataset_repo, repo_type="dataset", private=args.private, exist_ok=True)
        try:
            push_dataset_eval_yaml(args.dataset_repo)
            print(f"[ok] dataset eval.yaml uploaded")
        except Exception as exc:
            print(f"[warn] eval.yaml upload: {exc}", file=sys.stderr)
        print(f"[ok] dataset repo: https://huggingface.co/datasets/{args.dataset_repo}")

    if args.model_repo:
        ensure_model_repo(args.model_repo, private=args.private)
        readme = f"""---
license: mit
tags:
- uploadm8
- promo-targeting
- content-success
- tabular
datasets:
- {args.dataset_repo or 'your-org/promo-targeting-v1'}
---

# UploadM8 ML Hub Models

Models trained by the UploadM8 ML engine, sharing one dataset repo (separate splits):

- **Promo uplift** (`split: train`) — touchpoint conversion uplift.
  Eval: `.eval_results/uploadm8_promo_uplift.yaml`, metrics: `uploadm8_metrics.json`.
- **Content success / hottest topic** (`split: content_success`) — learns which
  topic, packaging, and upload-flow choices drive the strongest per-platform
  engagement (likes/comments/shares/views per video) and ranks the hottest content.
  Eval: `.eval_results/uploadm8_content_success.yaml`, metrics + ranked topics:
  `uploadm8_content_metrics.json`.

Evaluation metrics follow [Hugging Face eval results](https://huggingface.co/docs/hub/eval-results).
"""
        api.upload_file(
            path_or_fileobj=readme.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=args.model_repo,
            repo_type="model",
            commit_message="UploadM8 ML engine: model card scaffold",
        )
        print(f"[ok] model repo: https://huggingface.co/{args.model_repo}")

    if args.trackio_space:
        api.create_repo(
            args.trackio_space,
            repo_type="space",
            private=args.private,
            exist_ok=True,
            space_sdk="gradio",
        )
        for path_in_repo, content in (
            ("README.md", _SPACE_README),
            ("requirements.txt", _SPACE_REQUIREMENTS),
            ("app.py", _SPACE_APP),
        ):
            api.upload_file(
                path_or_fileobj=content.encode("utf-8"),
                path_in_repo=path_in_repo,
                repo_id=args.trackio_space,
                repo_type="space",
                commit_message="UploadM8 Trackio dashboard: bucket restore on startup",
            )
        project = (os.environ.get("TRACKIO_PROJECT") or "uploadm8-ml").strip()
        _configure_trackio_space(api, args.trackio_space, token=token, project=project)
        if args.sync_trackio:
            try:
                import trackio

                trackio.sync(
                    project=project,
                    space_id=args.trackio_space,
                    force=True,
                    bucket_id=_trackio_bucket_id(args.trackio_space),
                )
                print(f"[ok] synced local Trackio project '{project}' to bucket")
            except Exception as exc:
                print(f"[warn] trackio sync failed: {exc}", file=sys.stderr)
        print(f"[ok] space repo: https://huggingface.co/spaces/{args.trackio_space}")

    print("\nNext: set UM8_HF_DATASET_REPO, UM8_HF_MODEL_REPO, UM8_TRACKIO_SPACE_PATH in .env,")
    print("      UM8_ML_ENGINE_ENABLED=1, then restart the API.")


if __name__ == "__main__":
    main()
