#!/usr/bin/env python3
"""
Export a lean inference-only checkpoint (no optimizer) for deployment.

Usage:
  python scripts/export_inference_checkpoint.py
  python scripts/export_inference_checkpoint.py --input models/best_model.pth
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from services.model_utils import detect_model_type


def export_checkpoint(
    input_path: Path, output_path: Path, allow_unvalidated: bool = False
) -> None:
    print(f"Loading {input_path} ...")
    ckpt = torch.load(input_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        if not allow_unvalidated and (
            not ckpt.get("split_manifest") or not ckpt.get("test_metrics")
        ):
            raise ValueError(
                "Checkpoint lacks leakage-safe split provenance or held-out test metrics. "
                "Retrain with train.py, or pass --allow-unvalidated for research only."
            )
        state_dict = ckpt["model_state_dict"]
        lean = {
            "model_state_dict": state_dict,
            "model_type": ckpt.get("model_type")
            or detect_model_type(list(state_dict.keys())),
            "model_config": ckpt.get("model_config") or {},
            "source_checkpoint": str(input_path),
            "source_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
            "epoch": ckpt.get("epoch"),
            "seed": ckpt.get("seed"),
            "val_psnr": ckpt.get("val_psnr"),
            "val_ssim": ckpt.get("val_ssim"),
            "test_metrics": ckpt.get("test_metrics"),
            "split_manifest": ckpt.get("split_manifest"),
        }
    else:
        if not allow_unvalidated:
            raise ValueError(
                "Raw state dictionaries contain no validation provenance. "
                "Pass --allow-unvalidated for research only."
            )
        lean = {"model_state_dict": ckpt}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(lean, output_path)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Saved lean checkpoint to {output_path} ({size_mb:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "models" / "best_model.pth",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "models" / "deploy" / "model.pt",
    )
    parser.add_argument(
        "--allow-unvalidated",
        action="store_true",
        help="Export a checkpoint without held-out test provenance (research only)",
    )
    args = parser.parse_args()
    export_checkpoint(args.input, args.output, args.allow_unvalidated)


if __name__ == "__main__":
    main()
