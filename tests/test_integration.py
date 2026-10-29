"""
Full End-to-End System Integration and Quality Assurance Tests.
Guarding the Stones — Egocentric Heritage Protection AI.

Validates the entire pipeline from raw video creation to manifest ingestion,
feature caching, model packaging, inference, evaluation, and dashboard binding.
"""
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from src.config import (
    CANDIDATE_ACTION_CLASSES,
    SUPPORTED_VIDEO_EXTENSIONS,
    DatasetConfig,
)
from src.video_parser import extract_video_metadata
from src.dataset_manager import DatasetManager
from src.annotation_schema import (
    VideoAnnotation,
    export_json_schema,
    ANNOTATION_JSON_SCHEMA,
)
from src.preprocessing import VideoPreprocessor
from src.feature_extractor import FeatureExtractor
from src.feature_cache import FeatureCacheManager
from src.models import (
    ModelConfig,
    build_model,
    TemporalConv1DNet,
    TemporalPoolingBaseline,
    DEFAULT_ACTION_RISK_WEIGHTS,
)
from src.train import run_pretraining_checks, split_dataset_group_aware
from src.evaluate import evaluate_model_on_test_set, compute_expected_calibration_error
from src.inference import (
    InferenceEngine,
    PredictionResult,
    save_inference_package,
    export_predictions_json,
    export_predictions_csv,
)
from src.dashboard import create_dashboard, render_verdict_card


class TestFullSystemIntegration(unittest.TestCase):
    """Integration test suite executing the entire lifecycle on synthetic video fixtures."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.workspace = Path(cls.temp_dir.name)

        cls.raw_dir = cls.workspace / "raw"
        cls.raw_dir.mkdir(parents=True, exist_ok=True)
        cls.annotations_dir = cls.workspace / "annotations"
        cls.annotations_dir.mkdir(parents=True, exist_ok=True)
        cls.processed_dir = cls.workspace / "processed"
        cls.processed_dir.mkdir(parents=True, exist_ok=True)
        cls.eval_dir = cls.workspace / "evaluations"
        cls.eval_dir.mkdir(parents=True, exist_ok=True)
        cls.models_dir = cls.workspace / "models"
        cls.models_dir.mkdir(parents=True, exist_ok=True)

        # Create synthetic videos in raw directory (subfolders simulate carving and normal)
        cls.carving_dir = cls.raw_dir / "carving"
        cls.carving_dir.mkdir(exist_ok=True)
        cls.normal_dir = cls.raw_dir / "normal"
        cls.normal_dir.mkdir(exist_ok=True)

        cls.video_paths = []
        for i in range(3):
            p_c = cls.carving_dir / f"carving_synth_{i:02d}.mp4"
            cls._create_synthetic_video(p_c, num_frames=16, fps=20.0, fill_color=(50 + i * 40, 10, 10))
            cls.video_paths.append(p_c)

            p_n = cls.normal_dir / f"normal_synth_{i:02d}.mp4"
            cls._create_synthetic_video(p_n, num_frames=16, fps=20.0, fill_color=(10, 50 + i * 40, 10))
            cls.video_paths.append(p_n)

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    @staticmethod
    def _create_synthetic_video(path: Path, num_frames: int = 16, fps: float = 20.0, fill_color=(100, 100, 100)):
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, fps, (128, 128))
        for i in range(num_frames):
            frame = np.full((128, 128, 3), fill_value=fill_color, dtype=np.uint8)
            writer.write(frame)
        writer.release()

    def test_01_video_metadata_extraction(self):
        """Step 1: Verify OpenCV and container metadata extraction on synthetic clips."""
        v_path = self.video_paths[0]
        meta = extract_video_metadata(v_path)
        self.assertTrue(meta.is_readable)
        self.assertEqual(meta.validation_status, "valid")
        self.assertEqual(meta.format, ".mp4")
        self.assertEqual(meta.width, 128)
        self.assertEqual(meta.height, 128)
        self.assertEqual(meta.frame_count, 16)
        self.assertAlmostEqual(meta.fps, 20.0, places=1)
        self.assertGreater(len(meta.sha256), 0)

    def test_02_manifest_and_schema_validation(self):
        """Step 2: Generate dataset manifest and validate annotation schema integrity."""
        cfg = DatasetConfig(
            data_dir=self.raw_dir,
            annotations_dir=self.annotations_dir,
        )
        manager = DatasetManager(cfg)
        manifest_records = manager.build_manifest(target_dir=self.raw_dir)
        self.assertEqual(len(manifest_records), 6)

        m_paths = manager.export_manifest(manifest_records, out_base_path=self.annotations_dir / "manifest")
        self.assertTrue(Path(m_paths["jsonl"]).exists())
        self.assertTrue(Path(m_paths["csv"]).exists())

        # Test schema export and VideoAnnotation validation
        schema_path = self.annotations_dir / "schema.json"
        export_json_schema(schema_path)
        self.assertTrue(schema_path.exists())
        self.assertIn("properties", ANNOTATION_JSON_SCHEMA)

        annot = VideoAnnotation(
            video_id="VID_TEST_001",
            file_name="carving_synth_00.mp4",
            action_label="potential_scratching",
            camera_viewpoint="egocentric",
            is_egocentric=True,
            start_sec=0.0,
            end_sec=1.0,
            verified_by_human=True,
            annotator_id="test_annotator",
        )
        is_valid, errors = annot.validate(video_duration_sec=2.0)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
        self.assertEqual(annot.action_label, "potential_scratching")

    def test_03_preprocessing_and_feature_caching(self):
        """Step 3: Extract features and store in atomic NPZ cache."""
        preprocessor = VideoPreprocessor(num_frames=8, img_size=224)
        extractor = FeatureExtractor(backbone_name="mobilenet_v3_small", weights="none")

        cache_mgr = FeatureCacheManager(
            extractor=extractor,
            preprocessor=preprocessor,
            cache_dir=self.processed_dir,
        )

        for v_path in self.video_paths:
            meta = extract_video_metadata(v_path)
            rec, recomputed = cache_mgr.process_and_cache_video(
                video_id=v_path.name,
                video_path=v_path,
                sha256=meta.sha256,
                force_recompute=True,
            )
            self.assertTrue(recomputed)
            self.assertEqual(rec.per_frame_features.shape, (8, 576))
            self.assertEqual(rec.pooled_feature.shape, (1152,))

        cache_mgr.save()
        self.assertTrue(cache_mgr.cache_file.exists())
        self.assertEqual(len(cache_mgr.records), 6)

    def test_04_model_packaging_and_loading(self):
        """Step 4: Serialize deployment package and load via InferenceEngine."""
        m_cfg = ModelConfig(
            model_type="conv1d",
            backbone="mobilenet_v3_small",
            num_frames=8,
            class_names=sorted(list(CANDIDATE_ACTION_CLASSES)),
        )
        model = build_model(m_cfg)
        model.eval()

        pkg_path = self.models_dir / "guarding_model_pkg.pt"
        saved_pkg = save_inference_package(model, pkg_path)
        self.assertTrue(saved_pkg.exists())

        # Initialize InferenceEngine from package
        engine = InferenceEngine(model_path=pkg_path)
        self.assertEqual(engine.config.model_type, "conv1d")
        self.assertEqual(engine.config.backbone, "mobilenet_v3_small")
        self.assertEqual(len(engine.class_names), len(CANDIDATE_ACTION_CLASSES))

    def test_05_inference_and_temporal_evidence(self):
        """Step 5: Run single video and batch inference with temporal frame saliency."""
        pkg_path = self.models_dir / "guarding_model_pkg.pt"
        engine = InferenceEngine(model_path=pkg_path)

        # Single video with gallery frames
        res, gallery = engine.predict_video_with_frames(self.video_paths[0])
        self.assertEqual(res.status, "success")
        self.assertIn(res.predicted_label, CANDIDATE_ACTION_CLASSES)
        self.assertGreater(res.video_duration_sec, 0.0)
        self.assertEqual(len(res.sampled_frames), 8)
        self.assertIsNotNone(gallery)
        self.assertEqual(len(gallery), 8)

        # Batch prediction
        batch_rep = engine.predict_batch(self.video_paths)
        self.assertEqual(batch_rep.total_videos, 6)
        self.assertEqual(batch_rep.successful, 6)
        self.assertEqual(batch_rep.errors, 0)

        # JSON & CSV Export
        json_out = export_predictions_json(batch_rep, self.eval_dir / "integration_predictions.json")
        csv_out = export_predictions_csv(batch_rep, self.eval_dir / "integration_predictions.csv")
        self.assertTrue(json_out.exists())
        self.assertTrue(csv_out.exists())

    def test_06_model_evaluation_and_calibration(self):
        """Step 6: Run evaluation metrics and ECE calculation."""
        m_cfg = ModelConfig(
            model_type="conv1d",
            backbone="mobilenet_v3_small",
            input_dim=576,
            num_frames=8,
            num_classes=2,
            class_names=["normal", "carving"],
        )
        model = build_model(m_cfg)
        model.eval()

        # Synthetic test data
        rng = np.random.RandomState(42)
        X_test = rng.randn(10, 8, 576).astype(np.float32)
        y_test = rng.randint(0, 2, size=10)

        metrics = evaluate_model_on_test_set(
            model=model,
            X_test=X_test,
            y_test=y_test,
            class_names=["normal", "carving"],
            model_name="Integration_Conv1D",
            num_frames=8,
        )
        self.assertIsInstance(metrics.accuracy, float)
        self.assertIsInstance(metrics.expected_calibration_error, float)
        self.assertEqual(len(metrics.confusion_matrix), 2)
        self.assertGreater(metrics.throughput_clips_sec, 0.0)

    def test_07_dashboard_integration(self):
        """Step 7: Verify Gradio Blocks dashboard binds to the packaged inference engine."""
        pkg_path = self.models_dir / "guarding_model_pkg.pt"
        engine = InferenceEngine(model_path=pkg_path)
        app = create_dashboard(engine=engine)
        self.assertIsNotNone(app)

        # Test initial verdict rendering
        init_html = render_verdict_card(None, 0.35, 0.70, 0.40)
        self.assertIn("READY FOR VIDEO INGESTION", init_html)

    def test_08_cpu_only_operation_and_no_cuda_dependency(self):
        """Step 8: Verify model operates strictly on CPU without CUDA buffers."""
        pkg_path = self.models_dir / "guarding_model_pkg.pt"
        engine = InferenceEngine(model_path=pkg_path)

        for param in engine.model.parameters():
            self.assertEqual(param.device.type, "cpu")

        for buf in engine.model.buffers():
            self.assertEqual(buf.device.type, "cpu")

    def test_09_error_handling_graceful_failures(self):
        """Step 9: Verify missing model checkpoints and corrupt files do not crash system."""
        # 1. Missing checkpoint raises FileNotFoundError on engine init
        with self.assertRaises(FileNotFoundError):
            InferenceEngine(model_path=self.workspace / "missing_model.pt")

        # 2. Corrupt/empty video produces structured error result without unhandled crash
        engine = InferenceEngine()
        corrupt_video = self.workspace / "corrupt.mp4"
        corrupt_video.write_bytes(b"corrupt header bytes not valid video")

        res = engine.predict_video(corrupt_video)
        self.assertEqual(res.status, "error")
        self.assertEqual(res.verdict, "ERROR")
        self.assertIn(res.error_type, ("DECODING_ERROR", "CORRUPT_OR_UNREADABLE"))

    def test_10_reproducible_dataset_splitting(self):
        """Step 10: Verify group-aware splitting produces identical, disjoint splits across runs."""
        manifest_data = [
            {"video_path": f"v_{i}.mp4", "label": "normal" if i % 2 == 0 else "carving", "participant_id": f"p_{i % 3}"}
            for i in range(12)
        ]

        train_1, val_1, test_1 = split_dataset_group_aware(manifest_data, seed=42)
        train_2, val_2, test_2 = split_dataset_group_aware(manifest_data, seed=42)

        self.assertEqual(train_1, train_2)
        self.assertEqual(val_1, val_2)
        self.assertEqual(test_1, test_2)

        # Check disjointness
        s_train = {r["video_path"] for r in train_1}
        s_val = {r["video_path"] for r in val_1}
        s_test = {r["video_path"] for r in test_1}
        self.assertEqual(len(s_train.intersection(s_val)), 0)
        self.assertEqual(len(s_train.intersection(s_test)), 0)
        self.assertEqual(len(s_val.intersection(s_test)), 0)


if __name__ == "__main__":
    unittest.main()
