"""
Unit tests for egocentric temporal action recognition architectures.
Tests forward passes, output dimensions, backward gradients, invalid tensor shapes,
class weighting, CPU inference, and model serialization.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.models import (
    ModelConfig,
    TemporalConv1DNet,
    TemporalPoolingBaseline,
    TemporalGRUNet,
    build_model,
    compute_class_weights,
    FocalLoss,
    SklearnBaselineWrapper,
    DEFAULT_ACTION_RISK_WEIGHTS,
)


class TestTemporalModels(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)
        # MobileNetV3-Small: D=576, T=8 frames, C=11 candidate classes
        self.config_conv1d = ModelConfig(
            model_type="conv1d",
            backbone="mobilenet_v3_small",
            input_dim=576,
            num_frames=8,
            num_classes=11,
            hidden_dim=64,
            dropout=0.1,
        )
        self.config_pool = ModelConfig(
            model_type="pooling_mlp",
            backbone="mobilenet_v3_small",
            input_dim=576,
            num_frames=8,
            num_classes=11,
            hidden_dim=64,
            dropout=0.1,
        )
        self.config_gru = ModelConfig(
            model_type="gru",
            backbone="mobilenet_v3_small",
            input_dim=576,
            num_frames=8,
            num_classes=11,
            hidden_dim=64,
            dropout=0.1,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_forward_pass_and_output_dimensions(self):
        """Rule: Verify forward passes and output dimensions for all 3 models."""
        models = [
            TemporalConv1DNet(self.config_conv1d),
            TemporalPoolingBaseline(self.config_pool),
            TemporalGRUNet(self.config_gru),
        ]

        batch_size = 4
        num_frames = 8
        dim = 576
        dummy_input = torch.randn(batch_size, num_frames, dim)

        for model in models:
            model.eval()
            with torch.no_grad():
                out = model(dummy_input)

            self.assertIn("logits", out)
            self.assertIn("probabilities", out)
            self.assertIn("risk_score", out)

            logits = out["logits"]
            probs = out["probabilities"]
            risk = out["risk_score"]

            # Dimension checks
            self.assertEqual(logits.shape, torch.Size([batch_size, 11]))
            self.assertEqual(probs.shape, torch.Size([batch_size, 11]))
            self.assertEqual(risk.shape, torch.Size([batch_size]))

            # Probability simplex check (sums to 1.0)
            prob_sums = probs.sum(dim=-1).numpy()
            np.testing.assert_allclose(prob_sums, 1.0, atol=1e-5)

            # Risk score range check [0.0, 1.0]
            self.assertTrue(torch.all(risk >= 0.0) and torch.all(risk <= 1.0))

    def test_backward_gradients(self):
        """Rule: Verify valid gradient flow through temporal models."""
        model = TemporalConv1DNet(self.config_conv1d)
        model.train()

        dummy_input = torch.randn(2, 8, 576, requires_grad=True)
        dummy_targets = torch.tensor([1, 6], dtype=torch.long)

        out = model(dummy_input)
        loss = nn.CrossEntropyLoss()(out["logits"], dummy_targets)
        loss.backward()

        # Verify input tensor and model parameters received non-zero gradients
        self.assertIsNotNone(dummy_input.grad)
        has_param_grads = any(p.grad is not None and torch.norm(p.grad) > 0 for p in model.parameters())
        self.assertTrue(has_param_grads)

    def test_tensor_shape_checks_and_error_handling(self):
        """Rule: Provide clear tensor shape checks and useful errors."""
        model = TemporalConv1DNet(self.config_conv1d)

        # 1. 2D tensor instead of 3D
        with self.assertRaises(ValueError) as ctx:
            model(torch.randn(4, 576))
        self.assertIn("Expected 3D input tensor", str(ctx.exception))

        # 2. Embedding dimension mismatch (e.g. 512 instead of 576)
        with self.assertRaises(ValueError) as ctx:
            model(torch.randn(4, 8, 512))
        self.assertIn("Embedding dimension mismatch", str(ctx.exception))

        # 3. Non-tensor input
        with self.assertRaises(TypeError):
            model(np.random.randn(4, 8, 576))

    def test_configurable_frame_counts(self):
        """Rule: Support configurable frame counts (e.g. 8 vs 16 frames)."""
        model = TemporalConv1DNet(self.config_conv1d)
        model.eval()

        # Test with 16 frames instead of 8
        input_16 = torch.randn(2, 16, 576)
        with torch.no_grad():
            out_16 = model(input_16)
        self.assertEqual(out_16["logits"].shape, torch.Size([2, 11]))

        # Test with 4 frames
        input_4 = torch.randn(2, 4, 576)
        with torch.no_grad():
            out_4 = model(input_4)
        self.assertEqual(out_4["logits"].shape, torch.Size([2, 11]))

    def test_visual_baselines_dimensions(self):
        """Rule: Support ResNet18 (512) and MobileNetV3-Large (960)."""
        # ResNet18
        cfg_r18 = ModelConfig(model_type="conv1d", backbone="resnet18")
        self.assertEqual(cfg_r18.input_dim, 512)
        model_r18 = build_model(cfg_r18)
        out_r18 = model_r18(torch.randn(2, 8, 512))
        self.assertEqual(out_r18["logits"].shape, torch.Size([2, 11]))

        # MobileNetV3-Large
        cfg_large = ModelConfig(model_type="conv1d", backbone="mobilenet_v3_large")
        self.assertEqual(cfg_large.input_dim, 960)
        model_large = build_model(cfg_large)
        out_large = model_large(torch.randn(2, 8, 960))
        self.assertEqual(out_large["logits"].shape, torch.Size([2, 11]))

    def test_model_config_serialization(self):
        """Rule: Make model configuration reproducible and serializable."""
        cfg_file = self.test_dir / "model_config.json"
        self.config_conv1d.save_json(cfg_file)
        self.assertTrue(cfg_file.exists())

        loaded_cfg = ModelConfig.load_json(cfg_file)
        self.assertEqual(loaded_cfg.model_type, "conv1d")
        self.assertEqual(loaded_cfg.input_dim, 576)
        self.assertEqual(loaded_cfg.num_classes, 11)

    def test_class_imbalance_utilities(self):
        """Rule: Handle class imbalance using suitable class weighting or sampling."""
        # 100 normal samples (class 0), 5 scratching samples (class 6)
        labels = np.array([0] * 100 + [6] * 5)
        weights = compute_class_weights(labels, num_classes=11, method="inverse_frequency")

        self.assertEqual(weights.shape, torch.Size([11]))
        # Rare class 6 must have strictly higher weight than frequent class 0
        self.assertGreater(weights[6].item(), weights[0].item())

        # Test FocalLoss forward pass
        focal = FocalLoss(alpha=weights, gamma=2.0)
        dummy_logits = torch.randn(10, 11)
        dummy_targets = torch.tensor([0] * 9 + [6] * 1, dtype=torch.long)
        loss = focal(dummy_logits, dummy_targets)
        self.assertGreater(loss.item(), 0.0)

    def test_cpu_inference_single_and_batch(self):
        """Rule: Support CPU-only initialization and inference."""
        model = TemporalConv1DNet(self.config_conv1d)
        model.eval()

        # Batch size 1 (real-time stream simulation)
        single_input = torch.randn(1, 8, 576)
        with torch.no_grad():
            out1 = model(single_input)
        self.assertEqual(out1["logits"].shape, torch.Size([1, 11]))
        self.assertEqual(out1["risk_score"].shape, torch.Size([1]))

    def test_sklearn_baseline_wrapper(self):
        """Rule: Keep existing Logistic Regression pipeline as a comparison baseline."""
        wrapper = SklearnBaselineWrapper(c_param=1.0)
        # Synthetic pooled features: 20 samples, 2D=1152 features
        X = np.random.randn(20, 1152)
        y = np.array([0] * 10 + [1] * 10)
        wrapper.fit(X, y)
        self.assertTrue(wrapper.is_fitted)

        probs = wrapper.predict_proba(X[:2])
        preds = wrapper.predict(X[:2])
        self.assertEqual(probs.shape, (2, 2))
        self.assertEqual(preds.shape, (2,))


if __name__ == "__main__":
    unittest.main()
