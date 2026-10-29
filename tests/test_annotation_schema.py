"""
Unit tests for annotation schema validation, candidate classes, and timestamp constraints.
"""
import json
import tempfile
import unittest
from pathlib import Path

from src.config import CANDIDATE_ACTION_CLASSES
from src.annotation_schema import (
    VideoAnnotation,
    export_json_schema,
    ANNOTATION_JSON_SCHEMA,
)


class TestAnnotationSchema(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_all_11_candidate_classes_defined(self):
        expected_classes = {
            "normal_looking",
            "walking",
            "pointing",
            "photographing",
            "permitted_touching",
            "inspection",
            "potential_scratching",
            "potential_scraping",
            "potential_chipping",
            "potential_marking",
            "ambiguous_interaction",
        }
        self.assertEqual(CANDIDATE_ACTION_CLASSES, expected_classes)

    def test_valid_verified_annotation(self):
        ann = VideoAnnotation(
            video_id="VID_TEST_001",
            file_name="egocentric_scratching_01.mp4",
            action_label="potential_scratching",
            camera_viewpoint="egocentric",
            is_egocentric=True,
            start_sec=1.5,
            end_sec=5.0,
            watermark_detected=False,
            tool_used="key",
            target_surface="sandstone",
            verified_by_human=True,
            annotator_id="ANN_EXPERT_1",
            confidence=0.95,
        )
        is_valid, errors = ann.validate(video_duration_sec=10.0)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)

    def test_rejection_of_unverified_action_label(self):
        """Rule: Never assign labels automatically based on filenames without human verification."""
        ann = VideoAnnotation(
            video_id="VID_AUTO_001",
            file_name="carving 02.mp4",
            action_label="potential_scratching",  # Asserted label
            verified_by_human=False,              # Not human-verified
        )
        is_valid, errors = ann.validate()
        self.assertFalse(is_valid)
        self.assertTrue(any("cannot be asserted without human verification" in e for e in errors))

    def test_rejection_of_invalid_action_class(self):
        ann = VideoAnnotation(
            video_id="VID_BAD_001",
            file_name="test.mp4",
            action_label="random_unsupported_class",
            verified_by_human=True,
        )
        is_valid, errors = ann.validate()
        self.assertFalse(is_valid)
        self.assertTrue(any("Invalid action_label" in e for e in errors))

    def test_rejection_of_invalid_temporal_timestamps(self):
        # Case A: end_sec <= start_sec
        ann_reversed = VideoAnnotation(
            video_id="VID_TIME_001",
            file_name="test.mp4",
            action_label="walking",
            start_sec=5.0,
            end_sec=2.0,
            verified_by_human=True,
        )
        is_valid, errors = ann_reversed.validate(video_duration_sec=10.0)
        self.assertFalse(is_valid)
        self.assertTrue(any("strictly greater than start_sec" in e for e in errors))

        # Case B: end_sec exceeds clip duration
        ann_exceeds = VideoAnnotation(
            video_id="VID_TIME_002",
            file_name="test.mp4",
            action_label="walking",
            start_sec=1.0,
            end_sec=15.0,
            verified_by_human=True,
        )
        is_valid, errors = ann_exceeds.validate(video_duration_sec=10.0)
        self.assertFalse(is_valid)
        self.assertTrue(any("exceeds video duration" in e for e in errors))

        # Case C: negative start_sec
        ann_neg = VideoAnnotation(
            video_id="VID_TIME_003",
            file_name="test.mp4",
            action_label="walking",
            start_sec=-1.0,
            end_sec=5.0,
            verified_by_human=True,
        )
        is_valid, errors = ann_neg.validate(video_duration_sec=10.0)
        self.assertFalse(is_valid)
        self.assertTrue(any("cannot be negative" in e for e in errors))

    def test_viewpoint_conflict(self):
        ann = VideoAnnotation(
            video_id="VID_VIEW_001",
            file_name="test.mp4",
            action_label="pointing",
            camera_viewpoint="third_person",
            is_egocentric=True,  # Conflict
            verified_by_human=True,
        )
        is_valid, errors = ann.validate()
        self.assertFalse(is_valid)
        self.assertTrue(any("Conflict" in e for e in errors))

    def test_json_schema_export(self):
        out_schema = self.test_dir / "schema.json"
        export_json_schema(out_schema)
        self.assertTrue(out_schema.exists())
        with open(out_schema, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["title"], "EgocentricHeritageVandalismAnnotation")
        self.assertIn("action_label", data["required"])


if __name__ == "__main__":
    unittest.main()
