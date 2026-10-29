"""
Video preprocessing and uniform temporal frame sampling module.
Handles OpenCV decoding, timestamp extraction, resizing, and normalization for PyTorch backbones.
"""
from dataclasses import dataclass
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import cv2
import torch


# Standard ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class VideoDecodingError(Exception):
    """Raised when a video cannot be opened or decoded."""
    pass


@dataclass
class SampledClip:
    """Container for sampled frames, timestamps, and sampling metadata."""
    frames_rgb: np.ndarray          # Shape: (N, H, W, 3), uint8 [0, 255]
    tensor: torch.Tensor            # Shape: (N, 3, H, W), float32 normalized
    timestamps_sec: List[float]     # Exact timestamp in seconds for each frame
    frame_indices: List[int]        # 0-based original frame indices
    fps: float                      # Native or estimated frame rate
    total_frames_in_video: int      # Count of available frames in video
    duration_sec: float             # Total duration of clip


class VideoPreprocessor:
    """
    Decodes videos using OpenCV, extracts uniform temporal samples,
    preserves exact timestamps, and applies PyTorch-compatible preprocessing.
    """

    def __init__(
        self,
        num_frames: int = 8,
        img_size: int = 224,
        mean: np.ndarray = IMAGENET_MEAN,
        std: np.ndarray = IMAGENET_STD,
    ):
        if num_frames <= 0:
            raise ValueError(f"num_frames must be positive, got {num_frames}")
        if img_size <= 0:
            raise ValueError(f"img_size must be positive, got {img_size}")

        self.num_frames = num_frames
        self.img_size = img_size
        self.mean = mean
        self.std = std

    def get_config_dict(self) -> Dict[str, Any]:
        """Return dictionary representation of preprocessing hyperparameters."""
        return {
            "num_frames": self.num_frames,
            "img_size": self.img_size,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }

    def decode_and_sample(self, video_path: Path) -> SampledClip:
        """
        Open video file, extract uniform temporal frame samples, and compute timestamps.
        Handles short videos, variable frame rates, and missing metadata.
        """
        path_str = str(Path(video_path).resolve())
        if not os.path.exists(path_str):
            raise VideoDecodingError(f"Video file does not exist: {path_str}")

        if os.path.getsize(path_str) == 0:
            raise VideoDecodingError(f"Video file is 0 bytes (empty): {path_str}")

        cap = cv2.VideoCapture(path_str)
        if not cap.isOpened():
            raise VideoDecodingError(f"OpenCV failed to open video: {path_str}")

        # Probe metadata
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0 or np.isnan(fps):
            fps = 25.0  # Safe default assumption if unrecorded

        frames_bgr: List[np.ndarray] = []
        timestamps: List[float] = []
        indices: List[int] = []

        # Strategy 1: Direct seeking when frame count is valid and positive
        if total_frames >= self.num_frames:
            target_indices = np.linspace(0, total_frames - 1, self.num_frames).astype(int).tolist()
            seek_failed = False

            for idx in target_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, frame = cap.read()
                if ok and frame is not None:
                    frames_bgr.append(frame)
                    indices.append(idx)
                    # Timestamp from OpenCV POS_MSEC or index / fps
                    msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                    ts = (msec / 1000.0) if msec > 0 else (idx / fps)
                    timestamps.append(round(ts, 3))
                else:
                    seek_failed = True
                    break

            if seek_failed or len(frames_bgr) < self.num_frames:
                # Seek failed on this codec/container; fall back to sequential read
                frames_bgr.clear()
                timestamps.clear()
                indices.clear()

        # Strategy 2: Sequential reading (robust fallback for variable frame rates or unindexed streams)
        if len(frames_bgr) < self.num_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            all_frames: List[np.ndarray] = []
            frame_idx = 0
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                all_frames.append(frame)
                frame_idx += 1

            total_read = len(all_frames)
            if total_read == 0:
                cap.release()
                raise VideoDecodingError(f"Decoded 0 valid frames from video: {path_str}")

            total_frames = total_read
            # Uniform sample over all sequentially decoded frames
            if total_read >= self.num_frames:
                sample_indices = np.linspace(0, total_read - 1, self.num_frames).astype(int).tolist()
                for s_idx in sample_indices:
                    frames_bgr.append(all_frames[s_idx])
                    indices.append(s_idx)
                    timestamps.append(round(s_idx / fps, 3))
            else:
                # Short video handling: fewer frames than num_frames
                # Loop/replicate the available frames cyclically up to num_frames
                for i in range(self.num_frames):
                    s_idx = i % total_read
                    frames_bgr.append(all_frames[s_idx])
                    indices.append(s_idx)
                    timestamps.append(round(s_idx / fps, 3))

        cap.release()

        # Convert frames to RGB and resize
        frames_rgb_list: List[np.ndarray] = []
        for f in frames_bgr:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            if (rgb.shape[1], rgb.shape[0]) != (self.img_size, self.img_size):
                rgb = cv2.resize(rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            frames_rgb_list.append(rgb)

        frames_rgb_arr = np.stack(frames_rgb_list)  # (N, H, W, 3)

        # PyTorch normalization: (N, H, W, 3) -> (N, 3, H, W) float32 [0, 1] normalized
        x = frames_rgb_arr.astype(np.float32) / 255.0
        x = (x - self.mean) / self.std
        tensor = torch.from_numpy(x).permute(0, 3, 1, 2)

        duration = (total_frames / fps) if fps > 0 else 0.0

        return SampledClip(
            frames_rgb=frames_rgb_arr,
            tensor=tensor,
            timestamps_sec=timestamps,
            frame_indices=indices,
            fps=fps,
            total_frames_in_video=total_frames,
            duration_sec=round(duration, 3),
        )
