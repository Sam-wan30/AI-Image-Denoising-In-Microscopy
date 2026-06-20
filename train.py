#!/usr/bin/env python3
"""
Training script for CARE microscopy image denoising.

Uses Residual U-Net, L1+SSIM loss, LR scheduling, early stopping,
and saves best checkpoint by validation PSNR.
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from src.care_dataset_simple import CAREDatasetSimple
from src.unet_model import create_unet_model
from utils.data_splitting import grouped_holdout_split, grouped_kfold_indices
from utils.losses import DenoisingLoss
from utils.metrics import calculate_mae, calculate_mse, calculate_psnr, calculate_ssim

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable experiments."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_epoch_samples(model, dataset, device, epoch, save_dir, indices=None):
    """Save noisy | denoised | clean comparison for sample indices."""
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    indices = indices if indices is not None else [0]

    with torch.no_grad():
        for i in indices:
            noisy, clean = dataset[i]
            noisy_b = noisy.unsqueeze(0).to(device)
            pred = model(noisy_b).cpu().squeeze().numpy()
            noisy_np = noisy.squeeze().numpy()
            clean_np = clean.squeeze().numpy()

            panels = [
                (noisy_np * 255).astype(np.uint8),
                (np.clip(pred, 0, 1) * 255).astype(np.uint8),
                (clean_np * 255).astype(np.uint8),
            ]
            comparison = np.hstack(panels)
            cv2.imwrite(os.path.join(save_dir, f"epoch_{epoch:03d}_sample_{i}.png"), comparison)

    model.train()


def plot_training_curves(history, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, history["train_loss"], "b-", label="Train")
    axes[0].plot(epochs, history["val_loss"], "r-", label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, history["val_psnr"], "g-")
    axes[1].set_title("Validation PSNR (dB)")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True)

    axes[2].plot(epochs, history["val_ssim"], "m-")
    axes[2].set_title("Validation SSIM")
    axes[2].set_xlabel("Epoch")
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150)
    plt.close()


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total_loss = 0.0
    totals = {"psnr": 0.0, "ssim": 0.0, "mae": 0.0, "mse": 0.0}
    n_samples = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for noisy, clean in loader:
            noisy = noisy.to(device)
            clean = clean.to(device)

            if train:
                optimizer.zero_grad()

            pred = model(noisy)
            loss = criterion(pred, clean)

            if train:
                loss.backward()
                optimizer.step()

            batch_size = noisy.shape[0]
            total_loss += loss.item() * batch_size
            pred_np = pred.detach().cpu().numpy()
            clean_np = clean.detach().cpu().numpy()
            totals["psnr"] += float(
                np.sum(calculate_psnr(pred_np, clean_np, max_val=1.0, reduction="none"))
            )
            totals["ssim"] += float(
                np.sum(calculate_ssim(pred_np, clean_np, max_val=1.0, reduction="none"))
            )
            totals["mae"] += float(
                np.sum(calculate_mae(pred_np, clean_np, reduction="none"))
            )
            totals["mse"] += float(
                np.sum(calculate_mse(pred_np, clean_np, reduction="none"))
            )
            n_samples += batch_size

    denominator = max(n_samples, 1)
    return {
        "loss": total_loss / denominator,
        **{name: value / denominator for name, value in totals.items()},
        "samples": n_samples,
    }


def train_model(
    model,
    train_loader,
    val_loader,
    train_dataset,
    device,
    num_epochs=50,
    learning_rate=1e-4,
    save_dir="models",
    log_dir="logs",
    early_stop_patience=10,
    sample_indices=None,
    overfit_mode=False,
    split_manifest=None,
    seed=42,
    model_config=None,
):
    os.makedirs(save_dir, exist_ok=True)
    sample_dir = os.path.join(save_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)

    writer = SummaryWriter(log_dir) if TENSORBOARD_AVAILABLE else None

    model = model.to(device)
    criterion = DenoisingLoss(l1_weight=0.7, ssim_weight=0.3, use_ssim=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    best_val_psnr = -1.0
    epochs_without_improvement = 0
    history = {
        "train_loss": [],
        "train_psnr": [],
        "val_loss": [],
        "val_psnr": [],
        "val_ssim": [],
        "val_mae": [],
        "val_mse": [],
    }

    print(f"Training on {device} | params: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(num_epochs):
        t0 = time.time()
        train_metrics = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        val_metrics = run_epoch(
            model, val_loader, criterion, None, device, train=False
        )

        scheduler.step(val_metrics["psnr"])

        history["train_loss"].append(train_metrics["loss"])
        history["train_psnr"].append(train_metrics["psnr"])
        for name in ("loss", "psnr", "ssim", "mae", "mse"):
            history[f"val_{name}"].append(val_metrics[name])

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1}/{num_epochs} ({elapsed:.1f}s) | "
            f"train_loss={train_metrics['loss']:.5f} val_loss={val_metrics['loss']:.5f} | "
            f"train_psnr={train_metrics['psnr']:.2f} val_psnr={val_metrics['psnr']:.2f} "
            f"val_ssim={val_metrics['ssim']:.4f} | lr={lr:.2e}"
        )

        if writer:
            writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
            writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
            writer.add_scalar("PSNR/val", val_metrics["psnr"], epoch)
            writer.add_scalar("SSIM/val", val_metrics["ssim"], epoch)
            writer.add_scalar("LR", lr, epoch)

        save_epoch_samples(
            model, train_dataset, device, epoch + 1, sample_dir, sample_indices
        )

        if val_metrics["psnr"] > best_val_psnr:
            best_val_psnr = val_metrics["psnr"]
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_type": (model_config or {}).get("model_type", "residual"),
                    "model_config": model_config or {},
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "seed": seed,
                    "split_manifest": split_manifest,
                    "val_loss": val_metrics["loss"],
                    "val_psnr": val_metrics["psnr"],
                    "val_ssim": val_metrics["ssim"],
                    "val_mae": val_metrics["mae"],
                    "val_mse": val_metrics["mse"],
                    "train_loss": train_metrics["loss"],
                },
                os.path.join(save_dir, "best_model.pth"),
            )
            print(f"  -> Saved best model (val_psnr={val_metrics['psnr']:.2f})")
        else:
            epochs_without_improvement += 1
            print(
                f"  -> No PSNR improvement ({epochs_without_improvement}/{early_stop_patience})"
            )

        if epochs_without_improvement >= early_stop_patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    if writer:
        writer.close()

    plot_training_curves(history, save_dir)
    Path(save_dir, "training_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    return history, best_val_psnr


def build_loaders(
    data_dir,
    batch_size,
    val_split,
    test_split,
    augment,
    seed,
    max_samples=None,
    overfit_n=None,
    cv_folds=0,
    cv_fold=0,
    image_size=256,
):
    target_size = (image_size, image_size)
    train_source = CAREDatasetSimple(
        root_dir=data_dir, image_size=target_size, augment=augment
    )
    eval_source = CAREDatasetSimple(
        root_dir=data_dir, image_size=target_size, augment=False
    )
    pairs = train_source.image_pairs

    if overfit_n is not None:
        n = min(overfit_n, len(train_source))
        indices = list(range(n))
        split_indices = {"train": indices, "validation": indices, "test": []}
        print(f"OVERFIT MODE: using {n} image pairs for train and val")
    else:
        if max_samples is not None:
            pairs = pairs[: min(max_samples, len(pairs))]
            train_source.image_pairs = pairs
            eval_source.image_pairs = pairs
        if cv_folds:
            folds = grouped_kfold_indices(pairs, cv_folds, seed)
            if not 0 <= cv_fold < len(folds):
                raise ValueError(f"cv_fold must be between 0 and {len(folds) - 1}")
            split_indices = folds[cv_fold]
        else:
            split_indices = grouped_holdout_split(
                pairs, val_split, test_split, seed
            )

    train_dataset = Subset(train_source, split_indices["train"])
    val_dataset = Subset(eval_source, split_indices["validation"])
    test_dataset = (
        Subset(eval_source, split_indices["test"])
        if split_indices["test"]
        else None
    )

    manifest = {"seed": seed, "splits": {}}
    for split_name, indices in split_indices.items():
        manifest["splits"][split_name] = [
            {
                "noisy": Path(pairs[index][0]).name,
                "clean": Path(pairs[index][1]).name,
                "group": eval_source.group_id(index),
            }
            for index in indices
        ]
    print(
        "Split sizes: "
        + ", ".join(
            f"{name}={len(items)}" for name, items in manifest["splits"].items()
        )
    )

    generator = torch.Generator().manual_seed(seed)
    num_workers = 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    return train_loader, val_loader, test_loader, train_dataset, eval_source, manifest


def main():
    parser = argparse.ArgumentParser(description="Train microscopy denoising model")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_dir", type=str, default="models")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--test_split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv_folds", type=int, default=0)
    parser.add_argument("--cv_fold", type=int, default=0)
    parser.add_argument("--overfit", type=int, default=None, help="Train on N pairs only")
    parser.add_argument("--no_augment", action="store_true")
    parser.add_argument("--early_stop", type=int, default=10)
    parser.add_argument("--model_type", choices=("standard", "residual"), default="residual")
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, test_loader, train_dataset, full_dataset, manifest = build_loaders(
        args.data_dir,
        args.batch_size,
        args.val_split,
        args.test_split,
        augment=not args.no_augment and args.overfit is None,
        seed=args.seed,
        overfit_n=args.overfit,
        cv_folds=args.cv_folds,
        cv_fold=args.cv_fold,
        image_size=args.image_size,
    )

    os.makedirs(args.save_dir, exist_ok=True)
    Path(args.save_dir, "split_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    model_config = {
        "model_type": args.model_type,
        "in_channels": 1,
        "out_channels": 1,
        "base_channels": args.base_channels,
        "depth": args.depth,
        "bilinear": True,
        "image_size": args.image_size,
    }
    model = create_unet_model(**{k: v for k, v in model_config.items() if k != "image_size"})

    sample_indices = list(range(min(3, len(train_dataset))))
    history, best_psnr = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_dataset=train_dataset,
        device=device,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        early_stop_patience=args.early_stop,
        sample_indices=sample_indices,
        overfit_mode=args.overfit is not None,
        split_manifest=manifest,
        seed=args.seed,
        model_config=model_config,
    )

    if test_loader is not None:
        checkpoint = torch.load(
            os.path.join(args.save_dir, "best_model.pth"),
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        test_metrics = run_epoch(
            model,
            test_loader,
            DenoisingLoss(l1_weight=0.7, ssim_weight=0.3, use_ssim=True),
            None,
            device,
            train=False,
        )
        Path(args.save_dir, "test_metrics.json").write_text(
            json.dumps(test_metrics, indent=2), encoding="utf-8"
        )
        checkpoint["test_metrics"] = test_metrics
        torch.save(checkpoint, os.path.join(args.save_dir, "best_model.pth"))
        print(
            f"Held-out test: PSNR={test_metrics['psnr']:.2f} "
            f"SSIM={test_metrics['ssim']:.4f} MAE={test_metrics['mae']:.4f}"
        )

    print(f"\nTraining done. Best val PSNR: {best_psnr:.2f}")
    print(f"Best model: {os.path.join(args.save_dir, 'best_model.pth')}")


if __name__ == "__main__":
    main()
