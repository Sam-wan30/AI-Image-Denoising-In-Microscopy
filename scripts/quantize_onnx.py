#!/usr/bin/env python3
"""Quantize an ONNX checkpoint for memory-constrained CPU inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--image-size", type=int, default=128)
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(f"Model not found: {args.input}")

    quantize_dynamic(
        str(args.input),
        str(args.output),
        weight_type=QuantType.QInt8,
    )

    model = onnx.load(args.output)
    for value_info in (model.graph.input[0], model.graph.output[0]):
        dimensions = value_info.type.tensor_type.shape.dim
        dimensions[2].dim_value = args.image_size
        dimensions[3].dim_value = args.image_size
    # Intermediate shapes were inferred for the original 256px export. Runtime
    # infers them again from the smaller deployed input.
    model.graph.ClearField("value_info")
    onnx.save(model, args.output)

    print(
        f"Quantized model: {args.output} "
        f"({args.output.stat().st_size / 1024 / 1024:.1f} MB, "
        f"{args.image_size}x{args.image_size} input)"
    )


if __name__ == "__main__":
    main()
