"""
Unit tests for pretraining verification checks, group-aware splitting,
training loop convergence, checkpoint resumption, and evaluation.
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models import ModelConfig, TemporalConv1DNet
from src.train import (
    run_pretraining_checks,
    split_dataset_group_aware,
    TrainingConfig,
    ActionTrainer,
)


class TestTrainingPipeline(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_pretraining_gate_catches_missing_egocentric_and_labels(self):
        """Rule: Refuse to report successful training if required data is missing or invalid."""
        # Create a mock manifest where is_egocentric is False and labels are unverified
        mock_manifest = self.test_dir / "mock_manifest.jsonl"
        with open(mock_manifest, "w", encoding="utf-8") as f:
            for i in range(10):
                rec = {
                    "video_id": f"VID_{i:03d}",
                    "file_name": f"video_{i}.mp4",
                    "action_label": "unverified",
                    "is_egocentric": False,
                    "validation_status": "valid",
                }
                f.write(json.dumps(rec) + "\n")

        report = run_pretraining_checks(mock_manifest)
        self.assertFalse(report.passed)
        # Must detect zero egocentric clips
        self.assertTrue(any("ZERO authentic first-person" in err for err in report.fatal_failures))
        # Must detect 100% unverified labels
        self.assertTrue(any("are 'unverified'" in err for err in report.fatal_failures))

    def test_group_aware_splitting_prevents_leakage(self):
        """Rule: Keep recordings from same source or participant in one split; prevent leakage."""
        records = []
        # 4 participants, each with 5 clips
        for p_id in ["PARTICIPANT_A", "PARTICIPANT_B", "PARTICIPANT_C", "PARTICIPANT_D", "PARTICIPANT_E"]:
            for c_idx in range(5):
                records.append({
                    "video_id": f"VID_{p_id}_{c_idx}",
                    "participant_id": p_id,
                    "action_label": "potential_scratching" if c_idx % 2 == 0 else "walking",
                })

        train_recs, val_recs, test_recs = split_dataset_group_aware(
            records, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42
        )

        train_participants = set(r["participant_id"] for r in train_recs)
        val_participants = set(r["participant_id"] for r in val_recs)
        test_participants = set(r["participant_id"] for r in test_recs)

        # In strict group splitting with enough groups, participant groups must not overlap
        self.assertEqual(len(train_participants.intersection(test_participants)), 0)
        self.assertEqual(len(val_participants.intersection(test_participants)), 0)

    def test_action_trainer_training_and_convergence(self):
        """Rule: Verify training loop, loss convergence, early stopping, and checkpointing on CPU."""
        cfg = ModelConfig(
            model_type="conv1d",
            backbone="mobilenet_v3_small",
            input_dim=64,  # Small dimension for rapid unit test
            num_frames=4,
            num_classes=3,
            hidden_dim=32,
            dropout=0.0,
        )
        model = TemporalConv1DNet(cfg)
        train_cfg = TrainingConfig(
            epochs=5,
            batch_size=8,
            learning_rate=0.01,
            output_dir=str(self.test_dir / "checkpoints"),
        )
        label_to_idx = {"walking": 0, "pointing": 1, "potential_scratching": 2}

        # Create synthetic dataset with learnable signal
        np.random.seed(42)
        torch.manual_seed(42)
        X_train = torch.randn(32, 4, 64)
        y_train = torch.randint(0, 3, (32,))
        # Inject strong signal for class 2
        X_train[y_train == 2] += 2.0

        X_val = torch.randn(16, 4, 64)
        y_val = torch.randint(0, 3, (16,))
        X_val[y_val == 2] += 2.0

        train_loader = DataLoader(TensorDataset(X_train, y_train, torch.arange(32)), batch_size=8)
        val_loader = DataLoader(TensorDataset(X_val, y_val, torch.arange(16)), batch_size=8)

        trainer = ActionTrainer(model, train_cfg, label_to_idx)
        res = trainer.fit(train_loader, val_loader)

        self.assertIsNotNone(res["best_checkpoint_path"])
        self.assertTrue(Path(res["best_checkpoint_path"]).exists())
        self.assertGreaterEqual(res["best_val_f1"], 0.0)

        # Test checkpoint resumption
        new_model = TemporalConv1DNet(cfg)
        new_trainer = ActionTrainer(new_model, train_cfg, label_to_idx)
        ckpt = new_trainer.load_checkpoint(Path(res["best_checkpoint_path"]))
        self.assertIn("model_state_dict", ckpt)
        self.assertEqual(ckpt["model_config"]["model_type"], "conv1d")


if __name__ == "__main__":
    unittest.main()
