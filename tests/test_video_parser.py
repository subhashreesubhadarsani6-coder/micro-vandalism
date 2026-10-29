"""
Unit tests for video parsing, metadata extraction, and container validation.
"""
import struct
import tempfile
import unittest
from pathlib import Path

from src.video_parser import (
    extract_video_metadata,
    compute_sha256,
    parse_avi,
    parse_mp4_mov,
)


class TestVideoParser(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_non_existent_file(self):
        non_file = self.test_dir / "does_not_exist.mp4"
        meta = extract_video_metadata(non_file)
        self.assertFalse(meta.is_readable)
        self.assertEqual(meta.validation_status, "unreadable")
        self.assertIn("does not exist", meta.error_message)

    def test_empty_zero_byte_video(self):
        empty_file = self.test_dir / "empty_video.mp4"
        empty_file.touch()
        meta = extract_video_metadata(empty_file)
        self.assertFalse(meta.is_readable)
        self.assertEqual(meta.validation_status, "empty")
        self.assertEqual(meta.file_size_bytes, 0)
        self.assertIn("empty video", meta.error_message)

    def test_corrupt_header_video(self):
        corrupt_file = self.test_dir / "corrupt_video.mp4"
        with open(corrupt_file, "wb") as f:
            f.write(b"NOT_A_VALID_MP4_HEADER_GARBAGE_BYTES_1234567890")
        meta = extract_video_metadata(corrupt_file)
        self.assertFalse(meta.is_readable)
        self.assertEqual(meta.validation_status, "corrupt")

    def test_sha256_computation(self):
        test_file = self.test_dir / "sample.bin"
        with open(test_file, "wb") as f:
            f.write(b"Hello Heritage Guarding")
        import hashlib
        expected = hashlib.sha256(b"Hello Heritage Guarding").hexdigest()
        computed = compute_sha256(test_file)
        self.assertEqual(computed, expected)

    def test_synthetic_avi_riff_parsing(self):
        avi_file = self.test_dir / "synthetic.avi"
        # AVIMAINHEADER:
        # microsec=33333, max_bytes=100000, pad=0, flags=0x10, frames=300, init=0, streams=1, buf=100000, w=640, h=480
        avih_data = struct.pack(
            "<IIIIIIIIII",
            33333,   # 0: microsec
            100000,  # 4: max_bytes
            0,       # 8: padding
            0x10,    # 12: flags
            300,     # 16: total_frames
            0,       # 20: initial_frames
            1,       # 24: streams
            100000,  # 28: suggested_buffer
            640,     # 32: width
            480      # 36: height
        ) + (b"\x00" * 16) # reserved[4]
        avih_chunk = b"avih" + struct.pack("<I", len(avih_data)) + avih_data
        
        riff_body = b"AVI " + b"LIST" + struct.pack("<I", len(avih_chunk) + 4) + b"hdrl" + avih_chunk
        riff_header = b"RIFF" + struct.pack("<I", len(riff_body)) + riff_body

        with open(avi_file, "wb") as f:
            f.write(riff_header)

        dur, frames, w, h = parse_avi(avi_file)
        self.assertIsNotNone(dur)
        self.assertAlmostEqual(dur, 10.0, places=1)
        self.assertEqual(frames, 300)
        self.assertEqual(w, 640)
        self.assertEqual(h, 480)

        meta = extract_video_metadata(avi_file)
        self.assertTrue(meta.is_readable)
        self.assertEqual(meta.validation_status, "valid")
        self.assertEqual(meta.width, 640)
        self.assertEqual(meta.height, 480)


if __name__ == "__main__":
    unittest.main()
