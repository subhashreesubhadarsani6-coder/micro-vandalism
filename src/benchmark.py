"""
CPU Benchmarking Module for Video Preprocessing & Feature Extraction.
Guarding the Stones — Egocentric Heritage Protection AI.
"""
from dataclasses import dataclass
import os
from pathlib import Path
import time
from typing import List, Dict, Any, Optional

import numpy as np
import torch

from src.preprocessing import VideoPreprocessor
from src.feature_extractor import FeatureExtractor, SUPPORTED_BACKBONES


@dataclass
class BenchmarkResult:
    backbone: str
    num_frames: int
    n_clips: int
    avg_decode_time_ms: float
    avg_preprocess_time_ms: float
    avg_inference_time_ms: float
    avg_total_time_ms: float
    clips_per_sec: float
    effective_fps: float


class CPUBenchmark:
    """Runs repeatable CPU benchmarks across backbones and frame counts."""

    def __init__(self, sample_video_paths: List[Path], cpu_threads: Optional[int] = None):
        self.sample_paths = [Path(p).resolve() for p in sample_video_paths if Path(p).exists()]
        if not self.sample_paths:
            raise ValueError("No existing video paths provided for benchmark.")

        self.cpu_threads = cpu_threads or os.cpu_count() or 4
        torch.set_num_threads(self.cpu_threads)

    def run_benchmark(
        self,
        backbone: str = "resnet18",
        num_frames: int = 8,
        warmup_runs: int = 1,
    ) -> BenchmarkResult:
        preprocessor = VideoPreprocessor(num_frames=num_frames, img_size=224)
        extractor = FeatureExtractor(
            backbone_name=backbone,
            weights="none",
            cpu_threads=self.cpu_threads,
        )

        # Warmup pass
        for p in self.sample_paths[:warmup_runs]:
            try:
                clip = preprocessor.decode_and_sample(p)
                extractor.extract(clip.tensor)
            except Exception:
                pass

        decode_times = []
        preprocess_times = []
        inference_times = []
        total_times = []

        valid_clips = 0
        for p in self.sample_paths:
            try:
                # 1. Measure decoding
                t0 = time.perf_counter()
                clip = preprocessor.decode_and_sample(p)
                t_decode = time.perf_counter() - t0

                # 2. Measure tensor preprocessing (isolated)
                t1 = time.perf_counter()
                x = clip.frames_rgb.astype(np.float32) / 255.0
                x = (x - preprocessor.mean) / preprocessor.std
                t_tensor = torch.from_numpy(x).permute(0, 3, 1, 2)
                t_prep = time.perf_counter() - t1

                # 3. Measure model inference on CPU
                t2 = time.perf_counter()
                with torch.no_grad():
                    extractor.extract(t_tensor)
                t_infer = time.perf_counter() - t2

                total_time = t_decode + t_prep + t_infer

                decode_times.append(t_decode * 1000.0)
                preprocess_times.append(t_prep * 1000.0)
                inference_times.append(t_infer * 1000.0)
                total_times.append(total_time * 1000.0)
                valid_clips += 1

            except Exception as e:
                continue

        if valid_clips == 0:
            raise RuntimeError("Benchmark failed: zero clips could be successfully processed.")

        avg_decode = float(np.mean(decode_times))
        avg_prep = float(np.mean(preprocess_times))
        avg_infer = float(np.mean(inference_times))
        avg_total = float(np.mean(total_times))

        clips_sec = 1000.0 / avg_total if avg_total > 0 else 0.0
        effective_fps = clips_sec * num_frames

        return BenchmarkResult(
            backbone=backbone,
            num_frames=num_frames,
            n_clips=valid_clips,
            avg_decode_time_ms=round(avg_decode, 2),
            avg_preprocess_time_ms=round(avg_prep, 2),
            avg_inference_time_ms=round(avg_infer, 2),
            avg_total_time_ms=round(avg_total, 2),
            clips_per_sec=round(clips_sec, 2),
            effective_fps=round(effective_fps, 2),
        )


def run_full_benchmark_suite(
    sample_videos: List[Path],
    backbones: List[str] = ["mobilenet_v3_small", "resnet18", "mobilenet_v3_large"],
    frame_counts: List[int] = [8, 16],
) -> List[BenchmarkResult]:
    """Execute matrix of backbones and frame counts to compare latency."""
    bench = CPUBenchmark(sample_videos)
    results = []
    for bb in backbones:
        for nf in frame_counts:
            res = bench.run_benchmark(backbone=bb, num_frames=nf)
            results.append(res)
    return results
