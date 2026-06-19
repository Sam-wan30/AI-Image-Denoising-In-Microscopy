#!/usr/bin/env python3
"""Quantize an ONNX checkpoint for memory-constrained CPU inference."""

from __future__ import annotations

import argparse
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(f"Model not found: {args.input}")

    quantize_dynamic(
        str(args.input),
        str(args.output),
        weight_type=QuantType.QInt8,
    )
    print(
        f"Quantized model: {args.output} "
        f"({args.output.stat().st_size / 1024 / 1024:.1f} MB)"
    )


if __name__ == "__main__":
    main()
