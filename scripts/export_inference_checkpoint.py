#!/usr/bin/env python3
"""
Export a lean inference-only checkpoint (no optimizer) for deployment.

Usage:
  python scripts/export_inference_checkpoint.py
  python scripts/export_inference_checkpoint.py --input models/overfit/best_model.pth
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def export_checkpoint(input_path: Path, output_path: Path) -> None:
    print(f"Loading {input_path} ...")
    ckpt = torch.load(input_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        lean = {
            "model_state_dict": ckpt["model_state_dict"],
            "model_type": ckpt.get("model_type"),
        }
    else:
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
        default=ROOT / "models" / "overfit" / "best_model.pth",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "models" / "deploy" / "model.pt",
    )
    args = parser.parse_args()
    export_checkpoint(args.input, args.output)


if __name__ == "__main__":
    main()
