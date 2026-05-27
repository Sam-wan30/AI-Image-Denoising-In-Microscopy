#!/usr/bin/env python3
"""Export a trained PyTorch checkpoint to ONNX for deployment.

Usage:
  python scripts/export_to_onnx.py --input models/deploy/model.pt --output models/deploy/model.onnx
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, nargs=2, default=[256, 256])
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args()

    try:
        import torch
    except Exception as exc:
        print("PyTorch is required to run this script. Activate a Python env with torch.", file=sys.stderr)
        raise

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from services.model_utils import detect_model_type
    from src.unet_model import create_unet_model

    ckpt_path: Path = args.input
    out_path: Path = args.output
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(2)

    print(f"Loading checkpoint {ckpt_path} ...")
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    except Exception as exc:
        print(f"Failed to load checkpoint: {exc}", file=sys.stderr)
        raise

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        model_type = ckpt.get("model_type") or detect_model_type(list(state_dict.keys()))
    else:
        state_dict = ckpt
        model_type = detect_model_type(list(state_dict.keys()))

    keys = list(state_dict.keys())
    if model_type == "residual":
        first_conv_key = next(k for k in keys if "unet.inc.double_conv.0.weight" in k)
        last_conv_key = next(k for k in keys if "unet.outc.conv.weight" in k)
    else:
        first_conv_key = next(k for k in keys if "inc.double_conv.0.weight" in k)
        last_conv_key = next(k for k in keys if "outc.conv.weight" in k)

    in_channels = state_dict[first_conv_key].shape[1]
    out_channels = state_dict[last_conv_key].shape[0]

    model = create_unet_model(model_type=model_type, in_channels=in_channels, out_channels=out_channels)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"Warning: checkpoint compatibility issues: missing={len(missing)}, unexpected={len(unexpected)}")

    model.eval()

    H, W = args.image_size
    dummy = torch.randn(1, in_channels, H, W, dtype=torch.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Exporting ONNX to {out_path} (opset={args.opset}) ...")
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        export_params=True,
        opset_version=args.opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )

    print("ONNX export complete.")


if __name__ == "__main__":
    main()
