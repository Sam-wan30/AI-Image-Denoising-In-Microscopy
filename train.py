#!/usr/bin/env python3
"""
Training script for CARE microscopy image denoising.

Uses Residual U-Net, L1+SSIM loss, LR scheduling, early stopping,
and saves best checkpoint by validation PSNR.
"""

import argparse
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.care_dataset_simple import CAREDatasetSimple
from src.unet_model import create_unet_model
from utils.losses import DenoisingLoss
from utils.metrics import calculate_psnr, calculate_ssim

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None


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
    total_psnr = 0.0
    total_ssim = 0.0
    n_batches = 0

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

            total_loss += loss.item()
            total_psnr += calculate_psnr(pred, clean, max_val=1.0)
            total_ssim += calculate_ssim(pred, clean, max_val=1.0)
            n_batches += 1

    return (
        total_loss / max(n_batches, 1),
        total_psnr / max(n_batches, 1),
        total_ssim / max(n_batches, 1),
    )


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
    history = {"train_loss": [], "val_loss": [], "val_psnr": [], "val_ssim": []}

    print(f"Training on {device} | params: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(num_epochs):
        t0 = time.time()
        train_loss, train_psnr, train_ssim = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        val_loss, val_psnr, val_ssim = run_epoch(
            model, val_loader, criterion, None, device, train=False
        )

        scheduler.step(val_psnr)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_psnr"].append(val_psnr)
        history["val_ssim"].append(val_ssim)

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1}/{num_epochs} ({elapsed:.1f}s) | "
            f"train_loss={train_loss:.5f} val_loss={val_loss:.5f} | "
            f"train_psnr={train_psnr:.2f} val_psnr={val_psnr:.2f} val_ssim={val_ssim:.4f} | lr={lr:.2e}"
        )

        if writer:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("PSNR/val", val_psnr, epoch)
            writer.add_scalar("SSIM/val", val_ssim, epoch)
            writer.add_scalar("LR", lr, epoch)

        save_epoch_samples(
            model, train_dataset, device, epoch + 1, sample_dir, sample_indices
        )

        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_type": "residual",
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_psnr": val_psnr,
                    "val_ssim": val_ssim,
                    "train_loss": train_loss,
                },
                os.path.join(save_dir, "best_model.pth"),
            )
            print(f"  -> Saved best model (val_psnr={val_psnr:.2f})")
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
    return history, best_val_psnr


def build_loaders(data_dir, batch_size, val_split, augment, max_samples=None, overfit_n=None):
    full_dataset = CAREDatasetSimple(root_dir=data_dir, augment=augment)

    if overfit_n is not None:
        n = min(overfit_n, len(full_dataset))
        indices = list(range(n))
        train_dataset = Subset(full_dataset, indices)
        val_dataset = Subset(full_dataset, indices)
        print(f"OVERFIT MODE: using {n} image pairs for train and val")
    elif max_samples is not None:
        n = min(max_samples, len(full_dataset))
        full_dataset = Subset(full_dataset, list(range(n)))

        dataset_size = len(full_dataset)
        val_size = max(1, int(val_split * dataset_size))
        train_size = dataset_size - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
    else:
        dataset_size = len(full_dataset)
        val_size = max(1, int(val_split * dataset_size))
        train_size = dataset_size - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )

    num_workers = 0 if os.name != "nt" else 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, train_dataset, full_dataset


def main():
    parser = argparse.ArgumentParser(description="Train microscopy denoising model")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_dir", type=str, default="models")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--overfit", type=int, default=None, help="Train on N pairs only")
    parser.add_argument("--no_augment", action="store_true")
    parser.add_argument("--early_stop", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, train_dataset, full_dataset = build_loaders(
        args.data_dir,
        args.batch_size,
        args.val_split,
        augment=not args.no_augment and args.overfit is None,
        overfit_n=args.overfit,
    )

    model = create_unet_model(model_type="residual", in_channels=1, out_channels=1)

    sample_indices = list(range(min(3, len(full_dataset))))
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
    )

    print(f"\nTraining done. Best val PSNR: {best_psnr:.2f}")
    print(f"Best model: {os.path.join(args.save_dir, 'best_model.pth')}")


if __name__ == "__main__":
    main()
