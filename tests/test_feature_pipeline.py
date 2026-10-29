"""
Unit tests for video preprocessing, temporal frame sampling, feature extraction,
output dimensions, cache invalidation, and error handling.
"""
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import cv2
import torch

from src.preprocessing import VideoPreprocessor, VideoDecodingError
from src.feature_extractor import FeatureExtractor, PretrainedWeightsUnavailableError
from src.feature_cache import FeatureCacheManager, IncompatibleCacheError


def create_synthetic_video(
    out_path: Path,
    num_frames: int = 30,
    width: int = 320,
    height: int = 240,
    fps: float = 25.0,
) -> None:
    """Helper to generate a valid synthetic test video with colored moving bars."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    for i in range(num_frames):
        # Create an image with shifting color patterns
        img = np.zeros((height, width, 3), dtype=np.uint8)
        color = (int((i * 15) % 255), int((i * 25) % 255), int((i * 35) % 255))
        cv2.rectangle(img, (i * 5, 20), (i * 5 + 40, height - 20), color, -1)
        writer.write(img)
    writer.release()


class TestFeaturePipeline(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)
        self.sample_video = self.test_dir / "test_sample.mp4"
        create_synthetic_video(self.sample_video, num_frames=50, fps=25.0)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_uniform_frame_sampling_8_and_16(self):
        """Rule: Support configurable uniform sampling of 8 or 16 frames per clip."""
        # Test 8 frames
        prep8 = VideoPreprocessor(num_frames=8, img_size=224)
        clip8 = prep8.decode_and_sample(self.sample_video)
        self.assertEqual(len(clip8.frames_rgb), 8)
        self.assertEqual(clip8.tensor.shape, torch.Size([8, 3, 224, 224]))

        # Test 16 frames
        prep16 = VideoPreprocessor(num_frames=16, img_size=224)
        clip16 = prep16.decode_and_sample(self.sample_video)
        self.assertEqual(len(clip16.frames_rgb), 16)
        self.assertEqual(clip16.tensor.shape, torch.Size([16, 3, 224, 224]))

    def test_timestamp_preservation(self):
        """Rule: Preserve original timestamps of sampled frames."""
        prep = VideoPreprocessor(num_frames=8, img_size=224)
        clip = prep.decode_and_sample(self.sample_video)
        timestamps = clip.timestamps_sec

        self.assertEqual(len(timestamps), 8)
        # Verify timestamps are monotonically non-decreasing
        for i in range(len(timestamps) - 1):
            self.assertLessEqual(timestamps[i], timestamps[i + 1])
        # Verify timestamps stay within clip duration
        self.assertLessEqual(timestamps[-1], clip.duration_sec + 0.1)

    def test_short_video_handling(self):
        """Rule: Handle short videos gracefully (fewer frames than requested)."""
        short_video = self.test_dir / "short.mp4"
        create_synthetic_video(short_video, num_frames=3, fps=10.0)

        prep = VideoPreprocessor(num_frames=8, img_size=224)
        clip = prep.decode_and_sample(short_video)
        self.assertEqual(len(clip.frames_rgb), 8)
        self.assertEqual(clip.tensor.shape, torch.Size([8, 3, 224, 224]))

    def test_corrupt_and_empty_video_handling(self):
        """Rule: Handle unreadable files and decoding errors."""
        empty_video = self.test_dir / "empty.mp4"
        empty_video.touch()

        corrupt_video = self.test_dir / "corrupt.mp4"
        with open(corrupt_video, "wb") as f:
            f.write(b"GARBAGE_DATA_CORRUPT")

        prep = VideoPreprocessor(num_frames=8)
        with self.assertRaises(VideoDecodingError):
            prep.decode_and_sample(empty_video)

        with self.assertRaises(VideoDecodingError):
            prep.decode_and_sample(corrupt_video)

    def test_backbone_output_dimensions(self):
        """Rule: Support MobileNetV3-Small, ResNet18 and MobileNetV3-Large."""
        prep = VideoPreprocessor(num_frames=8, img_size=224)
        clip = prep.decode_and_sample(self.sample_video)

        # 1. ResNet18: per-frame 512, pooled 1024
        ext_resnet = FeatureExtractor(backbone_name="resnet18", weights="none")
        per_frame, pooled = ext_resnet.extract(clip.tensor)
        self.assertEqual(per_frame.shape, (8, 512))
        self.assertEqual(pooled.shape, (1024,))

        # 2. MobileNetV3-Small: per-frame 576, pooled 1152
        ext_m_small = FeatureExtractor(backbone_name="mobilenet_v3_small", weights="none")
        per_frame, pooled = ext_m_small.extract(clip.tensor)
        self.assertEqual(per_frame.shape, (8, 576))
        self.assertEqual(pooled.shape, (1152,))

        # 3. MobileNetV3-Large: per-frame 960, pooled 1920
        ext_m_large = FeatureExtractor(backbone_name="mobilenet_v3_large", weights="none")
        per_frame, pooled = ext_m_large.extract(clip.tensor)
        self.assertEqual(per_frame.shape, (8, 960))
        self.assertEqual(pooled.shape, (1920,))

    def test_pretrained_weight_protection(self):
        """Rule: If pretrained weights are not available locally, explain download requirement."""
        with self.assertRaises(PretrainedWeightsUnavailableError):
            FeatureExtractor(
                backbone_name="resnet18",
                weights="pretrained",
                allow_download=False,  # Strict protection
            )

    def test_cache_separation_and_invalidation(self):
        """Rule: Separate caches for different backbones and prevent stale reuse."""
        cache_dir = self.test_dir / "cache_store"
        prep8 = VideoPreprocessor(num_frames=8, img_size=224)
        ext_r18 = FeatureExtractor(backbone_name="resnet18", weights="none")

        cache_mgr1 = FeatureCacheManager(ext_r18, prep8, cache_dir=cache_dir)
        rec, was_computed = cache_mgr1.process_and_cache_video(
            video_id="VID_TEST_01",
            video_path=self.sample_video,
            sha256="fake_sha256_01",
        )
        self.assertTrue(was_computed)
        cache_mgr1.save()

        # Cache file must exist with correct naming
        expected_cache_name = "features_resnet18_8f_224.npz"
        self.assertTrue((cache_dir / expected_cache_name).exists())

        # Second load with same configuration should hit cache
        cache_mgr_hit = FeatureCacheManager(ext_r18, prep8, cache_dir=cache_dir)
        self.assertTrue(cache_mgr_hit.is_cached_and_current("VID_TEST_01", self.sample_video, "fake_sha256_01"))

        # Modified video (different SHA256) should trigger recompute
        self.assertFalse(cache_mgr_hit.is_cached_and_current("VID_TEST_01", self.sample_video, "different_sha"))

        # Incompatible backbone reading from same file path must raise IncompatibleCacheError
        ext_small = FeatureExtractor(backbone_name="mobilenet_v3_small", weights="none")
        cache_mgr_small = FeatureCacheManager(ext_small, prep8, cache_dir=cache_dir)
        # Manually point to resnet18 file to test header validation
        cache_mgr_small.cache_file = cache_dir / expected_cache_name
        with self.assertRaises(IncompatibleCacheError):
            cache_mgr_small.load()


if __name__ == "__main__":
    unittest.main()
