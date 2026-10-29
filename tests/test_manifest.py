"""
Unit tests for manifest generation, parsing, duplicate detection, and archive extraction safety.
"""
import csv
import json
import os
from pathlib import Path
import tempfile
import unittest
import zipfile

from src.config import DatasetConfig, get_dataset_dir
from src.dataset_manager import DatasetManager, DatasetRecord


class TestDatasetManagerAndManifest(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.data_dir = self.root / "data" / "raw"
        self.archive_path = self.root / "test_archive.zip"
        self.annotations_dir = self.root / "data" / "annotations"

        self.config = DatasetConfig(
            data_dir=self.data_dir,
            archive_path=self.archive_path,
            annotations_dir=self.annotations_dir,
        )
        self.manager = DatasetManager(self.config)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_safe_archive_extraction_no_overwrite(self):
        """Rule: Support extracting archive into configurable directory without overwriting existing files."""
        # Create a mock zip with 2 videos
        with zipfile.ZipFile(self.archive_path, "w") as zf:
            zf.writestr("test_video_1.mp4", b"SAMPLE_VIDEO_CONTENT_1")
            zf.writestr("test_video_2.mp4", b"SAMPLE_VIDEO_CONTENT_2")

        # Pre-create test_video_1 with DIFFERENT content in destination
        self.data_dir.mkdir(parents=True, exist_ok=True)
        existing_file = self.data_dir / "test_video_1.mp4"
        with open(existing_file, "wb") as f:
            f.write(b"PRE_EXISTING_ORIGINAL_CONTENT")

        # Run extraction
        res = self.manager.extract_archive()
        self.assertTrue(res["success"])
        self.assertEqual(res["extracted_count"], 1) # only test_video_2
        self.assertEqual(res["skipped_count"], 1)   # test_video_1 skipped

        # Verify test_video_1 was NOT overwritten
        with open(existing_file, "rb") as f:
            content = f.read()
        self.assertEqual(content, b"PRE_EXISTING_ORIGINAL_CONTENT")

        # Verify test_video_2 was extracted
        self.assertTrue((self.data_dir / "test_video_2.mp4").exists())

    def test_duplicate_detection_by_hash(self):
        """Rule: Detect potential duplicate clips."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        # Create two identical files with different names
        v1 = self.data_dir / "clip_a.mp4"
        v2 = self.data_dir / "clip_b.mp4"
        with open(v1, "wb") as f:
            f.write(b"IDENTICAL_CONTENT_FOR_HASH_TEST")
        with open(v2, "wb") as f:
            f.write(b"IDENTICAL_CONTENT_FOR_HASH_TEST")

        records = self.manager.build_manifest(target_dir=self.data_dir)
        self.assertEqual(len(records), 2)

        statuses = [r.validation_status for r in records]
        self.assertIn("duplicate", statuses)

    def test_manifest_export_and_parsing(self):
        """Rule: Create dataset manifest in CSV or JSONL format and test parsing."""
        record = DatasetRecord(
            video_id="VID_TEST_12345",
            file_path="data/raw/sample.mp4",
            file_name="sample.mp4",
            format=".mp4",
            file_size_bytes=1024,
            sha256="abc123456789",
            duration_sec=12.5,
            frame_count=300,
            fps=24.0,
            width=1920,
            height=1080,
            action_label="walking",
            camera_viewpoint="egocentric",
            is_egocentric=True,
            watermark_detected=False,
            validation_status="valid",
            source_archive="test.zip",
            notes="Parsed correctly",
        )

        paths = self.manager.export_manifest([record])
        jsonl_path = Path(paths["jsonl"])
        csv_path = Path(paths["csv"])

        self.assertTrue(jsonl_path.exists())
        self.assertTrue(csv_path.exists())

        # Test parsing JSONL
        with open(jsonl_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 1)
        parsed_json = json.loads(lines[0])
        self.assertEqual(parsed_json["video_id"], "VID_TEST_12345")
        self.assertEqual(parsed_json["duration_sec"], 12.5)
        self.assertEqual(parsed_json["action_label"], "walking")
        self.assertTrue(parsed_json["is_egocentric"])

        # Test parsing CSV
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["video_id"], "VID_TEST_12345")
        self.assertEqual(float(rows[0]["duration_sec"]), 12.5)

    def test_summary_and_problem_detection(self):
        """Rule: Provide clear summary of class counts and dataset problems."""
        rec_empty = DatasetRecord(
            video_id="VID_EMPTY",
            file_path="empty.mp4",
            file_name="empty.mp4",
            format=".mp4",
            file_size_bytes=0,
            sha256="",
            duration_sec=None,
            frame_count=None,
            fps=None,
            width=None,
            height=None,
            action_label="unverified",
            camera_viewpoint="stock_footage",
            is_egocentric=False,
            watermark_detected=True,
            validation_status="empty",
        )
        summary = self.manager.generate_summary([rec_empty])
        self.assertEqual(summary["total_videos"], 1)
        self.assertEqual(summary["watermark_count"], 1)
        self.assertEqual(summary["egocentric_count"], 0)
        self.assertTrue(any("zero-byte (empty)" in p for p in summary["detected_problems"]))
        self.assertTrue(any("Zero clips possess a validated first-person" in p for p in summary["detected_problems"]))

    def test_missing_labels_detection(self):
        """Rule: Test that missing/unverified action labels are properly flagged."""
        rec_unlabeled = DatasetRecord(
            video_id="VID_UNLABELED",
            file_path="sample.mp4",
            file_name="sample.mp4",
            format=".mp4",
            file_size_bytes=1000,
            sha256="abc",
            duration_sec=10.0,
            frame_count=240,
            fps=24.0,
            width=1920,
            height=1080,
            action_label="unverified",
            camera_viewpoint="unknown",
            is_egocentric=False,
            watermark_detected=False,
            validation_status="valid",
        )
        summary = self.manager.generate_summary([rec_unlabeled])
        self.assertEqual(summary["action_label_breakdown"].get("unverified"), 1)
        self.assertTrue(any("All action labels are 'unverified'" in p for p in summary["detected_problems"]))

    def test_invalid_videos_handling(self):
        """Rule: Test detection of invalid, unreadable, and corrupt video files."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        # Create an unreadable 0-byte video
        empty_v = self.data_dir / "zero_byte.mp4"
        empty_v.touch()

        # Create a corrupt garbage video
        corrupt_v = self.data_dir / "corrupt.mkv"
        with open(corrupt_v, "wb") as f:
            f.write(b"NOT_A_VIDEO")

        records = self.manager.build_manifest(target_dir=self.data_dir)
        status_map = {r.file_name: r.validation_status for r in records}
        
        self.assertEqual(status_map["zero_byte.mp4"], "empty")
        self.assertEqual(status_map["corrupt.mkv"], "corrupt")

        summary = self.manager.generate_summary(records)
        self.assertTrue(any("zero-byte (empty)" in p for p in summary["detected_problems"]))
        self.assertTrue(any("unreadable/corrupt files" in p for p in summary["detected_problems"]))

    def test_env_var_path_override(self):
        """Rule: Configurable through environment variable."""
        custom_path = self.root / "custom_env_data"
        os.environ["GUARDING_DATASET_DIR"] = str(custom_path)
        resolved = get_dataset_dir()
        self.assertEqual(resolved, custom_path.resolve())
        del os.environ["GUARDING_DATASET_DIR"]


if __name__ == "__main__":
    unittest.main()
