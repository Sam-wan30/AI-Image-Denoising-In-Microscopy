from __future__ import annotations

import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from scripts.export_inference_checkpoint import export_checkpoint
from services.denoiser import DenoiserService
from src.care_dataset_simple import CAREDatasetSimple
from train import build_loaders, run_epoch
from utils.data_splitting import (
    grouped_holdout_split,
    grouped_kfold_indices,
    specimen_group_id,
)
from utils.losses import DenoisingLoss
from utils.metrics import calculate_psnr


class ReliabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = CAREDatasetSimple("data", augment=False)

    def test_grouped_split_is_reproducible_and_disjoint(self):
        first = grouped_holdout_split(self.dataset.image_pairs, 0.2, 0.15, 42)
        second = grouped_holdout_split(self.dataset.image_pairs, 0.2, 0.15, 42)
        self.assertEqual(first, second)

        group_sets = {}
        for split_name, indices in first.items():
            group_sets[split_name] = {
                specimen_group_id(self.dataset.image_pairs[index][0])
                for index in indices
            }
        self.assertTrue(group_sets["train"].isdisjoint(group_sets["validation"]))
        self.assertTrue(group_sets["train"].isdisjoint(group_sets["test"]))
        self.assertTrue(group_sets["validation"].isdisjoint(group_sets["test"]))

    def test_grouped_cross_validation_covers_each_sample_once(self):
        folds = grouped_kfold_indices(self.dataset.image_pairs, 5, 42)
        validation_indices = [
            index for fold in folds for index in fold["validation"]
        ]
        self.assertEqual(sorted(validation_indices), list(range(len(self.dataset))))

    def test_validation_dataset_has_no_augmentation(self):
        train_loader, val_loader, _, _, _, _ = build_loaders(
            "data", 4, 0.2, 0.15, True, 42
        )
        self.assertTrue(train_loader.dataset.dataset.augment)
        self.assertFalse(val_loader.dataset.dataset.augment)

    def test_ssim_loss_backpropagates(self):
        prediction = torch.rand(2, 1, 32, 32, requires_grad=True)
        target = torch.rand(2, 1, 32, 32)
        loss = DenoisingLoss()(prediction, target)
        loss.backward()
        self.assertIsNotNone(prediction.grad)
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertGreater(float(prediction.grad.abs().sum()), 0.0)

    def test_training_epoch_reports_each_sample(self):
        noisy = torch.rand(3, 1, 16, 16)
        clean = torch.rand(3, 1, 16, 16)
        loader = DataLoader(TensorDataset(noisy, clean), batch_size=2)
        model = torch.nn.Sequential(torch.nn.Conv2d(1, 1, 1), torch.nn.Sigmoid())
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        metrics = run_epoch(
            model,
            loader,
            DenoisingLoss(),
            optimizer,
            torch.device("cpu"),
            train=True,
        )
        self.assertEqual(metrics["samples"], 3)
        self.assertTrue(np.isfinite(metrics["loss"]))

    def test_unvalidated_checkpoint_export_is_blocked(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "source.pt"
            output = Path(directory) / "output.pt"
            torch.save({"model_state_dict": {"weight": torch.ones(1)}}, source)
            with self.assertRaisesRegex(ValueError, "lacks leakage-safe"):
                export_checkpoint(source, output)

    def test_square_chw_metric_layout(self):
        image = np.ones((1, 32, 32), dtype=np.float32) * 0.5
        self.assertAlmostEqual(calculate_psnr(image, image), 120.0, places=4)

    def test_invalid_upload_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "valid supported image"):
            DenoiserService.validate_upload(b"not an image")

        buffer = BytesIO()
        Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(buffer, "PNG")
        image = DenoiserService.validate_upload(buffer.getvalue())
        self.assertEqual(image.size, (8, 8))


if __name__ == "__main__":
    unittest.main()
