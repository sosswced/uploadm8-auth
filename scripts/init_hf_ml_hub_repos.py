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
sdk_version: 4.44.1
python_version: "3.10"
app_file: app.py
pinned: false
license: mit
---

# UploadM8 + Trackio

This Space is reserved for experiment dashboards. Deploy Trackio from the
[Trackio project](https://github.com/tracking-ai/trackio) or replace `app.py` with your own
Gradio app, then set `TRACKIO_SPACE_ID` / `UM8_TRACKIO_SPACE_URL` in UploadM8.
"""

# Gradio 4.44 imports HfFolder from huggingface_hub; hub 1.0+ removed it. HF Spaces often ship hub 1.x
# while README sdk_version keeps Gradio 4.x — pin hub <1 so oauth imports succeed.
_SPACE_REQUIREMENTS = """gradio==4.44.1
huggingface_hub>=0.26.0,<1.0.0
"""

_SPACE_APP = '''"""Minimal Gradio app so the Space builds; replace with Trackio or your UI."""
import gradio as gr

with gr.Blocks(title="UploadM8 Trackio") as demo:
    gr.Markdown(
        "## UploadM8 — Trackio\\n\\n"
        "Placeholder Space. When you are ready, deploy Trackio here or wire "
        "`UM8_TRACKIO_SPACE_URL` in UploadM8 to this Space URL."
    )
'''


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
    p.add_argument("--private", action="store_true", help="Create private repos.")
    args = p.parse_args()

    token = _hf_write_token()
    if not token:
        _die("HF_TOKEN or HUGGING_FACE_HUB_TOKEN is required (write token).")

    if not args.dataset_repo and not args.trackio_space:
        _die(
            "Provide --dataset-repo and/or --trackio-space, or set "
            "UM8_HF_DATASET_REPO / UM8_TRACKIO_SPACE_PATH in the environment."
        )

    from huggingface_hub import HfApi

    api = HfApi(token=token)

    if args.dataset_repo:
        api.create_repo(args.dataset_repo, repo_type="dataset", private=args.private, exist_ok=True)
        print(f"[ok] dataset repo: https://huggingface.co/datasets/{args.dataset_repo}")

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
            )
        print(f"[ok] space repo: https://huggingface.co/spaces/{args.trackio_space}")

    print("\nNext: set UM8_HF_DATASET_REPO / UM8_TRACKIO_SPACE_PATH (or *_URL) in .env and restart the API.")


if __name__ == "__main__":
    main()
