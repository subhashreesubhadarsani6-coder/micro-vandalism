"""
Unit tests for Gradio Interactive CPU Dashboard & Heritage Security Interface.
Guarding the Stones — Egocentric Heritage Protection AI.
"""
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.config import CANDIDATE_ACTION_CLASSES
from src.models import ModelConfig
from src.inference import (
    InferenceEngine,
    PredictionResult,
    SampledFrameEvidence,
)
from src.dashboard import (
    create_dashboard,
    render_verdict_card,
    render_performance_card,
    render_timeline_summary,
    render_video_info_card,
    HIST_HEADERS,
)


class TestDashboardComponents(unittest.TestCase):
    """Test suite verifying Gradio dashboard cards, construction, and callbacks."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)

        # Create a synthetic MP4 video
        self.sample_video_path = self.test_dir / "synthetic_dash.mp4"
        self._create_synthetic_video(self.sample_video_path, num_frames=16, fps=24.0)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_synthetic_video(self, path: Path, num_frames: int = 16, fps: float = 24.0):
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, fps, (128, 128))
        for i in range(num_frames):
            frame = np.full((128, 128, 3), fill_value=(i * 15) % 256, dtype=np.uint8)
            writer.write(frame)
        writer.release()

    def _make_dummy_result(
        self,
        predicted_label: str = "normal_looking",
        confidence: float = 0.85,
        risk_score: float = 0.10,
        status: str = "success",
    ) -> PredictionResult:
        return PredictionResult(
            video_path=str(self.sample_video_path),
            status=status,
            predicted_label=predicted_label,
            confidence=confidence,
            risk_score=risk_score,
            action_category="benign_activity" if "normal" in predicted_label else "potential_damage",
            verdict="NORMAL" if risk_score < 0.35 else "POTENTIAL VANDALISM",
            uncertainty_flag=confidence < 0.40,
            probabilities={predicted_label: confidence},
            peak_evidence_timestamp_sec=0.50,
            peak_evidence_frame_idx=12,
            sampled_frames=[
                SampledFrameEvidence(
                    sample_index=0,
                    frame_index=0,
                    timestamp_sec=0.0,
                    attention_weight=0.125,
                    is_peak_evidence=False,
                ),
                SampledFrameEvidence(
                    sample_index=1,
                    frame_index=12,
                    timestamp_sec=0.50,
                    attention_weight=0.350,
                    is_peak_evidence=True,
                ),
            ],
            video_duration_sec=2.5,
            video_fps=24.0,
            video_total_frames=60,
            decode_and_preprocess_ms=45.0,
            inference_ms=15.0,
            total_latency_ms=60.0,
        )

    def test_render_verdict_card_states(self):
        """Verify HTML card rendering across initial, error, normal, and alert states."""
        # 1. Initial State (no video)
        card_init = render_verdict_card(None, 0.35, 0.70, 0.40)
        self.assertIn("READY FOR VIDEO INGESTION", card_init)

        # 2. Error State
        err_res = PredictionResult(
            video_path="",
            status="error",
            predicted_label="unknown",
            confidence=0.0,
            risk_score=0.0,
            action_category="unknown",
            verdict="ERROR",
            uncertainty_flag=True,
            probabilities={},
            peak_evidence_timestamp_sec=None,
            peak_evidence_frame_idx=None,
            sampled_frames=[],
            video_duration_sec=0.0,
            video_fps=0.0,
            video_total_frames=0,
            decode_and_preprocess_ms=0.0,
            inference_ms=0.0,
            total_latency_ms=0.0,
            error_type="CORRUPT_VIDEO",
            error_message="Cannot decode frames.",
        )
        card_err = render_verdict_card(err_res, 0.35, 0.70, 0.40)
        self.assertIn("Video Processing Error", card_err)
        self.assertIn("CORRUPT_VIDEO", card_err)

        # 3. Normal State
        res_norm = self._make_dummy_result(predicted_label="normal_looking", confidence=0.88, risk_score=0.05)
        card_norm = render_verdict_card(res_norm, 0.35, 0.70, 0.40)
        self.assertIn("NORMAL", card_norm)
        self.assertIn("CONFIDENT", card_norm)

        # 4. Potential Vandalism Alert State
        res_alert = self._make_dummy_result(predicted_label="potential_scratching", confidence=0.85, risk_score=0.88)
        card_alert = render_verdict_card(res_alert, 0.35, 0.70, 0.40)
        self.assertIn("POTENTIAL VANDALISM", card_alert)

    def test_render_performance_and_timeline_cards(self):
        """Verify performance and timeline cards contain latency, FPS, and timestamps."""
        res = self._make_dummy_result()
        perf = render_performance_card(res)
        self.assertIn("CPU Execution Profile", perf)
        self.assertIn("60.0 ms", perf)

        timeline = render_timeline_summary(res)
        self.assertIn("Temporal Sequence Summary", timeline)
        self.assertIn("2.50 s", timeline)
        self.assertIn("0.50 s", timeline)

    def test_dashboard_blocks_creation(self):
        """Verify Gradio Blocks compiles and initializes cleanly."""
        engine = InferenceEngine(
            config=ModelConfig(model_type="conv1d", backbone="mobilenet_v3_small", num_frames=8)
        )
        app = create_dashboard(engine=engine)
        self.assertIsNotNone(app)
        # Check that title is configured
        self.assertIn("Guarding the Stones", app.title)

    def test_render_video_info_card(self):
        """Verify video info card rendering for uninitialized and evaluated states."""
        # Uninitialized state
        card_empty = render_video_info_card(None, None)
        self.assertIn("No active video loaded", card_empty)

        # Populated state
        res = self._make_dummy_result()
        from src.video_parser import VideoMetadata
        meta = VideoMetadata(
            file_path=str(self.sample_video_path),
            file_name="synthetic_dash.mp4",
            file_size_bytes=1024,
            format="mp4",
            sha256="abc123def456",
            duration_sec=2.5,
            frame_count=60,
            fps=24.0,
            width=128,
            height=128,
        )
        card_pop = render_video_info_card(res, meta)
        self.assertIn("synthetic_dash.mp4", card_pop)
        self.assertIn("128 × 128 px", card_pop)
        self.assertIn("2.50 s", card_pop)
        self.assertIn("mobilenet_v3_small", card_pop)
        self.assertIn("CONV1D", card_pop)

    def test_predict_video_with_frames_pipeline(self):
        """Verify predict_video_with_frames returns both PredictionResult and gallery images."""
        engine = InferenceEngine(
            config=ModelConfig(model_type="conv1d", backbone="mobilenet_v3_small", num_frames=8)
        )
        res, gallery = engine.predict_video_with_frames(self.sample_video_path)

        self.assertEqual(res.status, "success")
        self.assertIsNotNone(gallery)
        self.assertEqual(len(gallery), 8)

        # Inspect first gallery item: tuple of (np.ndarray, str)
        frame_rgb, caption = gallery[0]
        self.assertIsInstance(frame_rgb, np.ndarray)
        self.assertEqual(frame_rgb.shape, (224, 224, 3))
        self.assertIn("Sample 1/8", caption)
        self.assertIn("0.00s", caption)


if __name__ == "__main__":
    unittest.main()
