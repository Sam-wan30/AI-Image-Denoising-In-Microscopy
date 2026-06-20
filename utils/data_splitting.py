"""Leakage-safe group splitting for paired microscopy images."""

from __future__ import annotations

import hashlib
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image

ImagePair = tuple[str, str]


def specimen_group_id(path: str | Path) -> str:
    """Group frames and laser-power variants from one specimen session."""
    stem = Path(path).stem
    stem = re.sub(r"_img_\d+$", "", stem, flags=re.IGNORECASE)
    return re.sub(
        r"_laser_power_\d+_\d+$", "", stem, flags=re.IGNORECASE
    ).lower()


def grouped_holdout_split(
    pairs: Sequence[ImagePair],
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, list[int]]:
    """Create deterministic train/validation/test splits by specimen session."""
    if not pairs:
        raise ValueError("Cannot split an empty dataset")
    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction and test_fraction must be >= 0 and sum to < 1")

    groups: dict[str, list[int]] = defaultdict(list)
    for index, (noisy_path, _clean_path) in enumerate(pairs):
        groups[specimen_group_id(noisy_path)].append(index)

    group_names = sorted(groups)
    if len(group_names) < 3 and test_fraction > 0:
        raise ValueError("At least three specimen groups are required for a test split")

    random.Random(seed).shuffle(group_names)
    val_count = max(1, round(len(group_names) * val_fraction)) if val_fraction else 0
    test_count = max(1, round(len(group_names) * test_fraction)) if test_fraction else 0
    if val_count + test_count >= len(group_names):
        raise ValueError("Not enough specimen groups for the requested split fractions")

    test_groups = set(group_names[:test_count])
    val_groups = set(group_names[test_count : test_count + val_count])
    split_indices = {"train": [], "validation": [], "test": []}
    for group_name, indices in groups.items():
        split = (
            "test"
            if group_name in test_groups
            else "validation"
            if group_name in val_groups
            else "train"
        )
        split_indices[split].extend(indices)

    for indices in split_indices.values():
        indices.sort()
    assert_no_target_leakage(pairs, split_indices)
    return split_indices


def grouped_kfold_indices(
    pairs: Sequence[ImagePair], n_splits: int, seed: int
) -> list[dict[str, list[int]]]:
    """Return balanced, deterministic specimen-group cross-validation folds."""
    groups: dict[str, list[int]] = defaultdict(list)
    for index, (noisy_path, _clean_path) in enumerate(pairs):
        groups[specimen_group_id(noisy_path)].append(index)
    if n_splits < 2 or n_splits > len(groups):
        raise ValueError(f"n_splits must be between 2 and {len(groups)}")

    group_items = list(groups.items())
    random.Random(seed).shuffle(group_items)
    group_items.sort(key=lambda item: len(item[1]), reverse=True)
    fold_groups: list[list[tuple[str, list[int]]]] = [[] for _ in range(n_splits)]
    fold_sizes = [0] * n_splits
    for item in group_items:
        fold_index = min(range(n_splits), key=lambda index: fold_sizes[index])
        fold_groups[fold_index].append(item)
        fold_sizes[fold_index] += len(item[1])

    all_indices = set(range(len(pairs)))
    folds = []
    for fold in fold_groups:
        validation = sorted(index for _name, indices in fold for index in indices)
        train = sorted(all_indices.difference(validation))
        split = {"train": train, "validation": validation, "test": []}
        assert_no_target_leakage(pairs, split)
        folds.append(split)
    return folds


def assert_no_target_leakage(
    pairs: Sequence[ImagePair], split_indices: dict[str, Iterable[int]]
) -> None:
    """Raise when byte-identical clean targets occur in different splits."""
    owners: dict[str, str] = {}
    for split_name, indices in split_indices.items():
        for index in indices:
            clean_path = Path(pairs[index][1])
            with Image.open(clean_path) as image:
                pixels = image.convert("I").tobytes()
                identity = f"{image.size}:{image.mode}:".encode() + pixels
            digest = hashlib.sha256(identity).hexdigest()
            previous = owners.setdefault(digest, split_name)
            if previous != split_name:
                raise ValueError(
                    f"Clean-target leakage detected between {previous} and {split_name}: "
                    f"{clean_path.name}"
                )
