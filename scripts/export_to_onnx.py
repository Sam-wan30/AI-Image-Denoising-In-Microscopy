#!/usr/bin/env python3
"""Export a validated PyTorch denoiser to ONNX and verify inference parity."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.model_utils import detect_model_type
from src.unet_model import create_unet_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--allow-unvalidated", action="store_true")
    args = parser.parse_args()

    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Expected a checkpoint containing model_state_dict")
    if not args.allow_unvalidated and (
        not checkpoint.get("split_manifest") or not checkpoint.get("test_metrics")
    ):
        raise ValueError(
            "Checkpoint lacks split provenance or held-out test metrics. "
            "Use --allow-unvalidated only for research exports."
        )

    state_dict = checkpoint["model_state_dict"]
    model_type = checkpoint.get("model_type") or detect_model_type(list(state_dict))
    model_config = dict(checkpoint.get("model_config") or {})
    first_key = (
        "unet.inc.double_conv.0.weight"
        if model_type == "residual"
        else "inc.double_conv.0.weight"
    )
    last_key = (
        "unet.outc.conv.weight"
        if model_type == "residual"
        else "outc.conv.weight"
    )
    model = create_unet_model(
        model_type=model_type,
        in_channels=state_dict[first_key].shape[1],
        out_channels=state_dict[last_key].shape[0],
        base_channels=int(model_config.get("base_channels", 64)),
        depth=int(model_config.get("depth", 4)),
        bilinear=bool(model_config.get("bilinear", True)),
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    torch.manual_seed(int(checkpoint.get("seed") or 42))
    sample = torch.rand(1, state_dict[first_key].shape[1], args.image_size, args.image_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        sample,
        args.output,
        export_params=True,
        opset_version=args.opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )

    with torch.inference_mode():
        torch_output = model(sample).numpy()
    session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    onnx_output = session.run(
        None, {session.get_inputs()[0].name: sample.numpy().astype(np.float32)}
    )[0]
    max_abs_error = float(np.max(np.abs(torch_output - onnx_output)))
    if max_abs_error > 1e-4:
        raise RuntimeError(f"ONNX parity check failed: max_abs_error={max_abs_error}")

    metadata = {
        "source_checkpoint": str(args.input),
        "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "onnx_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "model_type": model_type,
        "model_config": model_config,
        "image_size": args.image_size,
        "opset": args.opset,
        "max_abs_parity_error": max_abs_error,
        "seed": checkpoint.get("seed"),
        "validation": {
            "psnr": checkpoint.get("val_psnr"),
            "ssim": checkpoint.get("val_ssim"),
        },
        "test_metrics": checkpoint.get("test_metrics"),
        "validated_export": bool(
            checkpoint.get("split_manifest") and checkpoint.get("test_metrics")
        ),
    }
    metadata_path = args.output.with_suffix(args.output.suffix + ".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {args.output}")
    print(f"Saved {metadata_path}")
    print(f"PyTorch/ONNX max absolute error: {max_abs_error:.3e}")


if __name__ == "__main__":
    main()
