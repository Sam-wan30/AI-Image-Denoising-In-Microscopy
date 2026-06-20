#!/usr/bin/env python3
"""Evaluate a denoising checkpoint against paired clean targets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import config
from services.denoiser import DenoiserService
from src.care_dataset_simple import CAREDatasetSimple
from utils.data_splitting import specimen_group_id
from utils.metrics import calculate_mae, calculate_mse, calculate_psnr, calculate_ssim
from utils.preprocessing import load_grayscale, normalize_image, resize_image


METRIC_NAMES = ("psnr", "ssim", "mae", "mse")


def image_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    return {
        "psnr": calculate_psnr(prediction, target, max_val=1.0),
        "ssim": calculate_ssim(prediction, target, max_val=1.0),
        "mae": calculate_mae(prediction, target),
        "mse": calculate_mse(prediction, target),
    }


def summarize(rows: list[dict], prefix: str) -> dict[str, dict[str, float]]:
    summary = {}
    for metric in METRIC_NAMES:
        values = np.asarray([row[f"{prefix}_{metric}"] for row in rows], dtype=np.float64)
        summary[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "median": float(np.median(values)),
        }
    return summary


def summarize_group_means(
    rows: list[dict], prefix: str
) -> dict[str, dict[str, float]]:
    """Confidence intervals over specimen means, not correlated frames."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["group"]].append(row)
    summary = {}
    for metric in METRIC_NAMES:
        values = np.asarray(
            [
                np.mean([row[f"{prefix}_{metric}"] for row in items])
                for items in groups.values()
            ],
            dtype=np.float64,
        )
        standard_error = (
            values.std(ddof=1) / math.sqrt(len(values)) if len(values) > 1 else 0.0
        )
        summary[metric] = {
            "group_mean": float(values.mean()),
            "group_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "ci95_low": float(values.mean() - 1.96 * standard_error),
            "ci95_high": float(values.mean() + 1.96 * standard_error),
            "groups": len(values),
        }
    return summary


def save_error_panel(row: dict, output_dir: Path) -> None:
    noisy = cv2.imread(row["noisy_path"], cv2.IMREAD_GRAYSCALE)
    clean = cv2.imread(row["clean_path"], cv2.IMREAD_GRAYSCALE)
    prediction = cv2.imread(row["prediction_path"], cv2.IMREAD_GRAYSCALE)
    target_size = (clean.shape[1], clean.shape[0])
    noisy = resize_image(noisy, target_size)
    prediction = resize_image(prediction, target_size)
    error = cv2.normalize(
        cv2.absdiff(prediction, clean), None, 0, 255, cv2.NORM_MINMAX
    )
    panel = np.hstack([noisy, prediction, clean, error])
    cv2.imwrite(str(output_dir / f"error_{Path(row['file']).stem}.png"), panel)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/model_validation"))
    parser.add_argument("--save-worst", type=int, default=5)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = args.output_dir / "predictions"
    prediction_dir.mkdir(exist_ok=True)

    dataset = CAREDatasetSimple(str(args.data_dir), augment=False)
    selected_names = None
    evaluation_scope = "all_pairs_retrospective"
    if args.manifest:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        selected_names = {
            item["noisy"] for item in manifest["splits"].get(args.split, [])
        }
        if not selected_names:
            raise ValueError(f"Manifest split is empty: {args.split}")
        evaluation_scope = f"manifest:{args.split}"

    config.MODEL_PATH = args.model
    service = DenoiserService()
    rows = []
    for noisy_path_text, clean_path_text in dataset.image_pairs:
        noisy_path = Path(noisy_path_text)
        clean_path = Path(clean_path_text)
        if selected_names is not None and noisy_path.name not in selected_names:
            continue

        noisy_image = Image.open(noisy_path).convert("RGB")
        prediction_u8 = service.denoise(noisy_image, mode="unet")
        prediction_path = prediction_dir / noisy_path.name
        Image.fromarray(prediction_u8).save(prediction_path)

        clean_native = normalize_image(load_grayscale(clean_path))
        comparison_size = (clean_native.shape[1], clean_native.shape[0])
        prediction = normalize_image(resize_image(prediction_u8, comparison_size))
        noisy = normalize_image(
            resize_image(load_grayscale(noisy_image), comparison_size)
        )
        clean = clean_native
        model_values = image_metrics(prediction, clean)
        baseline_values = image_metrics(noisy, clean)
        row = {
            "file": noisy_path.name,
            "group": specimen_group_id(noisy_path),
            "noisy_path": str(noisy_path),
            "clean_path": str(clean_path),
            "prediction_path": str(prediction_path),
        }
        row.update({f"model_{name}": value for name, value in model_values.items()})
        row.update({f"baseline_{name}": value for name, value in baseline_values.items()})
        row.update(
            {
                f"delta_{name}": model_values[name] - baseline_values[name]
                for name in METRIC_NAMES
            }
        )
        rows.append(row)

    if not rows:
        raise ValueError("No evaluation pairs selected")

    csv_fields = [key for key in rows[0] if not key.endswith("_path")]
    with (args.output_dir / "per_image_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in csv_fields} for row in rows)

    group_rows: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        group_rows[row["group"]].append(row)
    report = {
        "model": str(args.model),
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "scope": evaluation_scope,
        "pair_count": len(rows),
        "group_count": len(group_rows),
        "model_metrics": summarize(rows, "model"),
        "model_group_level_metrics": summarize_group_means(rows, "model"),
        "noisy_input_baseline": summarize(rows, "baseline"),
        "baseline_group_level_metrics": summarize_group_means(rows, "baseline"),
        "improvement_counts": {
            "psnr": sum(row["delta_psnr"] > 0 for row in rows),
            "ssim": sum(row["delta_ssim"] > 0 for row in rows),
            "total": len(rows),
        },
        "per_group": {
            group: {
                "count": len(items),
                "model_metrics": summarize(items, "model"),
                "baseline_metrics": summarize(items, "baseline"),
            }
            for group, items in sorted(group_rows.items())
        },
        "classification_metrics": {
            "status": "not_applicable",
            "reason": "Denoising is paired image-to-image regression, not classification.",
        },
        "limitations": [
            "PSNR and SSIM require aligned clean reference images.",
            "Error panels localize failures but do not establish biological validity.",
            "Clinical or quantitative microscopy use requires independent external validation.",
        ],
    }
    (args.output_dir / "evaluation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    worst = sorted(rows, key=lambda row: row["delta_psnr"])[: args.save_worst]
    for row in worst:
        save_error_panel(row, args.output_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
