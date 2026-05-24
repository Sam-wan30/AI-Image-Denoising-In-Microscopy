#!/usr/bin/env python3
"""Simple ONNX dynamic quantization helper.

Usage: python scripts/quantize_onnx.py --input model.onnx --output model.quant.onnx
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def quantize(input_path: Path, output_path: Path) -> int:
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except Exception as exc:
        print("onnxruntime.quantization not available. Install onnxruntime>=1.15.0.", file=sys.stderr)
        return 2

    print(f"Quantizing {input_path} -> {output_path} (dynamic)")
    quantize_dynamic(str(input_path), str(output_path), weight_type=QuantType.QInt8)
    print("Quantization complete")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.output)

    if not inp.exists():
        print(f"Input file does not exist: {inp}", file=sys.stderr)
        sys.exit(1)

    code = quantize(inp, out)
    sys.exit(code)


if __name__ == "__main__":
    main()
