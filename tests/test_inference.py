"""
Unit tests for CPU Video Inference, Temporal Evidence, and Risk Assessment Engine.
Guarding the Stones — Egocentric Heritage Protection AI.
"""
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from src.config import CANDIDATE_ACTION_CLASSES
from src.models import (
    ModelConfig,
    TemporalConv1DNet,
    TemporalPoolingBaseline,
    DEFAULT_ACTION_RISK_WEIGHTS,
)
from src.inference import (
    InferenceEngine,
    PredictionResult,
    SampledFrameEvidence,
    BatchPredictionReport,
    compute_verdict,
    categorize_action_label,
    save_inference_package,
    export_predictions_json,
    export_predictions_csv,
)


class TestInferenceEngine(unittest.TestCase):
    """Test suite verifying inference loading, input validation, schemas, and thresholds."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)

        # Create a small valid synthetic MP4 file for testing
        self.sample_video_path = self.test_dir / "synthetic_test.mp4"
        self._create_synthetic_video(self.sample_video_path, num_frames=16, fps=24.0)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_synthetic_video(self, path: Path, num_frames: int = 16, fps: float = 24.0):
        """Generate a valid synthetic video file using OpenCV."""
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, fps, (128, 128))
        for i in range(num_frames):
            frame = np.full((128, 128, 3), fill_value=(i * 15) % 256, dtype=np.uint8)
            writer.write(frame)
        writer.release()

    def test_model_loading_and_package_export(self):
        """Verify model initialization, package export, and checkpoint loading."""
        cfg = ModelConfig(
            model_type="conv1d",
            backbone="mobilenet_v3_small",
            num_frames=8,
            class_names=sorted(list(CANDIDATE_ACTION_CLASSES)),
        )
        model = TemporalConv1DNet(cfg)

        # Export package
        pkg_path = self.test_dir / "test_model_pkg.pt"
        exported = save_inference_package(model, pkg_path)
        self.assertTrue(exported.exists())

        # Load engine from package
        engine = InferenceEngine(model_path=pkg_path)
        self.assertEqual(engine.config.model_type, "conv1d")
        self.assertEqual(engine.config.num_frames, 8)
        self.assertEqual(len(engine.class_names), len(CANDIDATE_ACTION_CLASSES))

    def test_input_validation_and_corrupt_files(self):
        """Verify graceful handling of missing, unsupported, and 0-byte corrupt files."""
        engine = InferenceEngine()

        # 1. Missing file
        res_missing = engine.predict_video(self.test_dir / "non_existent.mp4")
        self.assertEqual(res_missing.status, "error")
        self.assertEqual(res_missing.error_type, "FILE_NOT_FOUND")
        self.assertEqual(res_missing.verdict, "ERROR")

        # 2. Unsupported extension
        txt_path = self.test_dir / "notes.txt"
        txt_path.write_text("not a video")
        res_ext = engine.predict_video(txt_path)
        self.assertEqual(res_ext.status, "error")
        self.assertEqual(res_ext.error_type, "UNSUPPORTED_FORMAT")

        # 3. 0-byte empty file
        empty_mp4 = self.test_dir / "empty.mp4"
        empty_mp4.touch()
        res_empty = engine.predict_video(empty_mp4)
        self.assertEqual(res_empty.status, "error")
        self.assertEqual(res_empty.error_type, "EMPTY_FILE")

    def test_prediction_output_schema_and_types(self):
        """Verify all fields and schema in PredictionResult on real video inference."""
        engine = InferenceEngine(
            config=ModelConfig(model_type="conv1d", backbone="mobilenet_v3_small", num_frames=8)
        )
        res = engine.predict_video(self.sample_video_path)

        self.assertEqual(res.status, "success")
        self.assertIn(res.predicted_label, CANDIDATE_ACTION_CLASSES)
        self.assertIsInstance(res.confidence, float)
        self.assertGreaterEqual(res.confidence, 0.0)
        self.assertLessEqual(res.confidence, 1.0)
        self.assertIsInstance(res.risk_score, float)
        self.assertGreaterEqual(res.risk_score, 0.0)
        self.assertLessEqual(res.risk_score, 1.0)
        self.assertIn(res.action_category, ["benign_activity", "ambiguous_interaction", "potential_damage"])
        self.assertIn(res.verdict, ["NORMAL", "UNCERTAIN / HUMAN REVIEW", "POTENTIAL VANDALISM"])
        self.assertIsInstance(res.uncertainty_flag, bool)
        self.assertGreater(res.video_duration_sec, 0.0)
        self.assertGreater(res.video_fps, 0.0)
        self.assertGreater(res.total_latency_ms, 0.0)
        self.assertIn("Action recognition detects temporal movement patterns", res.disclaimer)

        # Dictionary serialization
        d = res.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(len(d["probabilities"]), len(CANDIDATE_ACTION_CLASSES))

    def test_timestamp_alignment_and_temporal_evidence(self):
        """Verify sampled frames have monotonic timestamps and valid peak evidence."""
        engine = InferenceEngine(
            config=ModelConfig(model_type="conv1d", backbone="mobilenet_v3_small", num_frames=8)
        )
        res = engine.predict_video(self.sample_video_path)

        self.assertEqual(len(res.sampled_frames), 8)

        # Check monotonic timestamps and frame indices
        prev_time = -1.0
        prev_frame = -1
        has_peak = False

        for f_ev in res.sampled_frames:
            self.assertGreaterEqual(f_ev.timestamp_sec, prev_time)
            self.assertGreaterEqual(f_ev.frame_index, prev_frame)
            self.assertGreaterEqual(f_ev.attention_weight, 0.0)
            if f_ev.is_peak_evidence:
                has_peak = True
                self.assertEqual(f_ev.timestamp_sec, res.peak_evidence_timestamp_sec)
                self.assertEqual(f_ev.frame_index, res.peak_evidence_frame_idx)
            prev_time = f_ev.timestamp_sec
            prev_frame = f_ev.frame_index

        self.assertTrue(has_peak)

    def test_verdict_and_risk_threshold_logic(self):
        """Verify business logic for review_threshold, alert_threshold, and uncertainty."""
        # 1. High confidence benign action with low risk -> NORMAL
        verdict, uncertain = compute_verdict(
            predicted_label="normal_looking",
            confidence=0.85,
            risk_score=0.10,
            review_threshold=0.35,
            alert_threshold=0.70,
            confidence_threshold=0.40,
        )
        self.assertEqual(verdict, "NORMAL")
        self.assertFalse(uncertain)

        # 2. Low confidence -> UNCERTAIN / HUMAN REVIEW with uncertain=True
        verdict_low, uncertain_low = compute_verdict(
            predicted_label="normal_looking",
            confidence=0.25,  # below 0.40
            risk_score=0.10,
            confidence_threshold=0.40,
        )
        self.assertEqual(verdict_low, "UNCERTAIN / HUMAN REVIEW")
        self.assertTrue(uncertain_low)

        # 3. High risk damage action above alert threshold -> POTENTIAL VANDALISM
        verdict_alert, uncertain_alert = compute_verdict(
            predicted_label="potential_scratching",
            confidence=0.80,
            risk_score=0.85,
            alert_threshold=0.70,
        )
        self.assertEqual(verdict_alert, "POTENTIAL VANDALISM")
        self.assertFalse(uncertain_alert)

        # 4. Ambiguous action -> UNCERTAIN / HUMAN REVIEW
        verdict_amb, uncertain_amb = compute_verdict(
            predicted_label="ambiguous_interaction",
            confidence=0.75,
            risk_score=0.45,
            review_threshold=0.35,
        )
        self.assertEqual(verdict_amb, "UNCERTAIN / HUMAN REVIEW")
        self.assertTrue(uncertain_amb)

    def test_batch_prediction_and_export(self):
        """Verify batch inference execution and JSON/CSV file generation."""
        # Create second synthetic video
        video2 = self.test_dir / "synthetic_2.mp4"
        self._create_synthetic_video(video2, num_frames=10, fps=20.0)

        engine = InferenceEngine(
            config=ModelConfig(model_type="conv1d", backbone="mobilenet_v3_small", num_frames=8)
        )
        report = engine.predict_batch([self.sample_video_path, video2])

        self.assertEqual(report.total_videos, 2)
        self.assertEqual(report.successful, 2)
        self.assertEqual(report.errors, 0)
        self.assertEqual(len(report.results), 2)
        self.assertGreater(report.average_latency_ms, 0.0)

        # Export JSON
        out_json = self.test_dir / "batch_report.json"
        p_j = export_predictions_json(report, out_json)
        self.assertTrue(p_j.exists())
        with open(p_j, "r", encoding="utf-8") as f:
            data_j = json.load(f)
        self.assertEqual(data_j["total_videos"], 2)

        # Export CSV
        out_csv = self.test_dir / "batch_report.csv"
        p_c = export_predictions_csv(report, out_csv)
        self.assertTrue(p_c.exists())
        with open(p_c, "r", encoding="utf-8") as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 3)  # header + 2 rows


if __name__ == "__main__":
    unittest.main()
