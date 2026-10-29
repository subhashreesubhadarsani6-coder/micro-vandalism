"""
Unit tests for model evaluation, calibration metrics (ECE, Brier Score),
confusion matrix generation, and CSV/JSON reporting.
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.models import ModelConfig, TemporalConv1DNet, TemporalPoolingBaseline, SklearnBaselineWrapper
from src.evaluate import (
    compute_expected_calibration_error,
    evaluate_model_on_test_set,
    export_evaluation_results,
    EvaluationMetrics,
)


class TestEvaluationPipeline(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_expected_calibration_error_calculation(self):
        """Rule: Verify Expected Calibration Error (ECE) calculation."""
        # Perfectly calibrated predictions: conf=1.0 when correct, 0.0 when incorrect
        y_true = np.array([1, 1, 0, 0])
        y_prob = np.array([
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ])
        ece_perfect = compute_expected_calibration_error(y_prob, y_true, n_bins=5)
        self.assertEqual(ece_perfect, 0.0)

        # Uncalibrated: completely overconfident wrong predictions
        y_prob_wrong = np.array([
            [1.0, 0.0], # True is 1, predicted 0 with conf 1.0
            [1.0, 0.0],
            [0.0, 1.0], # True is 0, predicted 1 with conf 1.0
            [0.0, 1.0],
        ])
        ece_wrong = compute_expected_calibration_error(y_prob_wrong, y_true, n_bins=5)
        self.assertGreater(ece_wrong, 0.5)

    def test_evaluate_model_on_test_set_pytorch(self):
        """Rule: Comprehensive evaluation on held-out test data for PyTorch model."""
        cfg = ModelConfig(
            model_type="conv1d",
            backbone="mobilenet_v3_small",
            input_dim=64,
            num_frames=8,
            num_classes=2,
            hidden_dim=32,
            class_names=["normal", "carving"],
        )
        model = TemporalConv1DNet(cfg)

        X_test = np.random.randn(20, 8, 64)
        y_test = np.array([0] * 10 + [1] * 10)
        class_names = ["normal", "carving"]

        metrics = evaluate_model_on_test_set(
            model=model,
            X_test=X_test,
            y_test=y_test,
            class_names=class_names,
            model_name="TemporalConv1DNet",
            backbone="mobilenet_v3_small",
            num_frames=8,
        )

        self.assertEqual(metrics.num_test_samples, 20)
        self.assertGreaterEqual(metrics.accuracy, 0.0)
        self.assertGreaterEqual(metrics.macro_f1, 0.0)
        self.assertIsNotNone(metrics.roc_auc)
        self.assertIsNotNone(metrics.pr_auc)
        self.assertIsNotNone(metrics.brier_score)
        self.assertIsNotNone(metrics.expected_calibration_error)
        self.assertEqual(len(metrics.confusion_matrix), 2)
        self.assertGreater(metrics.throughput_clips_sec, 0.0)
        self.assertGreaterEqual(metrics.latency_mean_ms, 0.0)

    def test_evaluate_model_on_test_set_sklearn(self):
        """Rule: Comprehensive evaluation for Scikit-learn Logistic Regression baseline."""
        wrapper = SklearnBaselineWrapper(c_param=1.0)
        X_train = np.random.randn(20, 128)
        y_train = np.array([0] * 10 + [1] * 10)
        wrapper.fit(X_train, y_train)

        X_test = np.random.randn(10, 128)
        y_test = np.array([0] * 5 + [1] * 5)
        class_names = ["normal", "carving"]

        metrics = evaluate_model_on_test_set(
            model=wrapper,
            X_test=X_test,
            y_test=y_test,
            class_names=class_names,
            model_name="LogisticRegression_Baseline",
            backbone="mobilenet_v3_small",
            num_frames=8,
        )

        self.assertEqual(metrics.num_test_samples, 10)
        self.assertIn("normal", metrics.per_class_metrics)
        self.assertIn("carving", metrics.per_class_metrics)

    def test_export_evaluation_results(self):
        """Rule: Save metrics in JSON and CSV formats."""
        m = EvaluationMetrics(
            model_name="TestModel",
            backbone="mobilenet_v3_small",
            num_frames=8,
            dataset_version="v1.0",
            num_test_samples=10,
            accuracy=0.80,
            balanced_accuracy=0.80,
            macro_f1=0.79,
            weighted_f1=0.79,
            macro_precision=0.80,
            macro_recall=0.80,
            roc_auc=0.85,
            pr_auc=0.82,
            brier_score=0.15,
            expected_calibration_error=0.08,
            confusion_matrix=[[4, 1], [1, 4]],
            class_support={"normal": 5, "carving": 5},
            per_class_metrics={
                "normal": {"precision": 0.8, "recall": 0.8, "f1": 0.8, "support": 5},
                "carving": {"precision": 0.8, "recall": 0.8, "f1": 0.8, "support": 5},
            },
            latency_mean_ms=1.2,
            latency_median_ms=1.1,
            latency_p95_ms=1.5,
            latency_p99_ms=1.8,
            throughput_clips_sec=150.0,
            effective_fps=1200.0,
            peak_memory_mb=0.5,
        )

        out_paths = export_evaluation_results([m], self.test_dir)
        self.assertTrue(Path(out_paths["json"]).exists())
        self.assertTrue(Path(out_paths["csv_summary"]).exists())
        self.assertTrue(Path(out_paths["csv_per_class"]).exists())

        # Verify JSON content
        with open(out_paths["json"], "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["model_name"], "TestModel")
        self.assertEqual(data[0]["accuracy"], 0.80)


if __name__ == "__main__":
    unittest.main()
