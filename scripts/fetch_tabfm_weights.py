"""Download TabFM's PyTorch weights and make them loadable by `tabfm`.

Why this script exists
-----------------------
As of ``tabfm==1.0.0``, the automatic download path in
``tabfm_v1_0_0_pytorch.load(checkpoint_path=None)`` looks for a file named
``pytorch_model.bin``. The files actually published on Hugging Face
(``google/tabfm-1.0.0-pytorch``) are ``model.safetensors``. Calling
``load()`` fresh, with no local override, therefore raises
``FileNotFoundError`` on a clean install — this is an upstream packaging
mismatch, not a mistake in your setup. Verified against the live Hugging
Face repo and the installed ``tabfm`` package source; see
docs/08-troubleshooting.md for the full writeup.

This script downloads the ``.safetensors`` weights via ``huggingface_hub``
(the same client `tabfm` itself uses) and re-saves them as
``pytorch_model.bin`` in the layout ``tabfm_v1_0_0_pytorch.load()`` expects.
The tensor names inside both formats are identical (verified by loading a
converted checkpoint with ``strict=True``), so no weight remapping is
required — this is a pure container-format conversion, not a model port.

Usage
-----
    uv run python scripts/fetch_tabfm_weights.py --task classification
    uv run python scripts/fetch_tabfm_weights.py --task regression
    uv run python scripts/fetch_tabfm_weights.py --task both   # default

Then point examples/notebooks at the result:
    export TABFM_CHECKPOINT_PATH=data/models/google-tabfm-1.0.0-pytorch
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from loguru import logger
from safetensors.torch import load_file

HF_REPO_ID = "google/tabfm-1.0.0-pytorch"
DEFAULT_OUTPUT_DIR = Path("data/models/google-tabfm-1.0.0-pytorch")


def convert_task(task: str, output_dir: Path) -> Path:
    """Downloads one task's safetensors weights and writes a pytorch_model.bin."""
    logger.info("Downloading TabFM {} weights from {} ...", task, HF_REPO_ID)
    snapshot_dir = Path(
        snapshot_download(repo_id=HF_REPO_ID, allow_patterns=[f"{task}/*"])
    )
    safetensors_path = snapshot_dir / task / "model.safetensors"
    if not safetensors_path.exists():
        raise FileNotFoundError(
            f"Expected {safetensors_path} after download. The upstream "
            "repo layout may have changed — check "
            f"https://huggingface.co/{HF_REPO_ID}."
        )

    state_dict = load_file(str(safetensors_path))

    task_output_dir = output_dir / task
    task_output_dir.mkdir(parents=True, exist_ok=True)
    bin_path = task_output_dir / "pytorch_model.bin"
    torch.save(state_dict, bin_path)
    logger.info("Wrote {} ({} tensors)", bin_path, len(state_dict))
    return bin_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=["classification", "regression", "both"],
        default="both",
        help="Which TabFM checkpoint(s) to fetch and convert.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Local directory to write converted checkpoints under.",
    )
    args = parser.parse_args()

    tasks = ["classification", "regression"] if args.task == "both" else [args.task]
    output_dir = Path(args.output_dir)
    for task in tasks:
        convert_task(task, output_dir)

    logger.info("Done. Set TABFM_CHECKPOINT_PATH={} to use these checkpoints.", output_dir)


if __name__ == "__main__":
    main()
