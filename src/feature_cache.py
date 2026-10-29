"""
Incremental, safe, and configuration-aware feature caching system.
Ensures separation across backbones, atomic writes to disk, and strict invalidation checks.
"""
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

from src.preprocessing import VideoPreprocessor, SampledClip, VideoDecodingError
from src.feature_extractor import FeatureExtractor


class IncompatibleCacheError(Exception):
    """Raised when loaded cache metadata does not match the active extraction pipeline."""
    pass


@dataclass
class CachedVideoRecord:
    """Individual video entry stored in the feature cache."""
    video_id: str
    file_path: str
    sha256: str
    mtime: float
    duration_sec: float
    timestamps_sec: List[float]
    pooled_feature: np.ndarray      # Shape: (pooled_dim,)
    per_frame_features: np.ndarray  # Shape: (N, per_frame_dim)


class FeatureCacheManager:
    """
    Manages loading, updating, and saving feature caches.
    Enforces atomic file writes and compatibility verification.
    """

    CACHE_SCHEMA_VERSION = "3.0"

    def __init__(
        self,
        extractor: FeatureExtractor,
        preprocessor: VideoPreprocessor,
        cache_dir: Optional[Path] = None,
    ):
        self.extractor = extractor
        self.preprocessor = preprocessor
        self.cache_dir = Path(cache_dir or "data/processed").resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.cache_file = self._compute_cache_filename()
        self.records: Dict[str, CachedVideoRecord] = {}
        self.metadata_header: Dict[str, Any] = self._build_header()

        # Load existing cache if present
        if self.cache_file.exists():
            self.load()

    def _compute_cache_filename(self) -> Path:
        """Construct isolated cache filename based on backbone, frames, and resolution."""
        backbone = self.extractor.backbone_name
        n_frames = self.preprocessor.num_frames
        res = self.preprocessor.img_size
        return self.cache_dir / f"features_{backbone}_{n_frames}f_{res}.npz"

    def _build_header(self) -> Dict[str, Any]:
        """Header metadata defining the exact extraction configuration."""
        return {
            "schema_version": self.CACHE_SCHEMA_VERSION,
            "backbone": self.extractor.backbone_name,
            "weights_mode": self.extractor.weights_mode,
            "per_frame_dim": self.extractor.per_frame_dim,
            "pooled_dim": self.extractor.pooled_dim,
            "pooling_method": "mean_max_concat",
            "num_frames": self.preprocessor.num_frames,
            "img_size": self.preprocessor.img_size,
            "mean": self.preprocessor.mean.tolist(),
            "std": self.preprocessor.std.tolist(),
        }

    def load(self) -> None:
        """
        Load and strictly validate existing cache file.
        Raises IncompatibleCacheError if configuration does not match.
        """
        if not self.cache_file.exists():
            return

        try:
            data = np.load(self.cache_file, allow_pickle=True)
            if "header_json" not in data:
                raise IncompatibleCacheError("Cache missing configuration header.")

            header = json.loads(str(data["header_json"]))

            # Verify crucial parameters for compatibility
            mismatches = []
            for key in ["backbone", "num_frames", "img_size", "pooled_dim"]:
                expected = self.metadata_header.get(key)
                actual = header.get(key)
                if expected != actual:
                    mismatches.append(f"{key}: expected '{expected}', found '{actual}'")

            if mismatches:
                raise IncompatibleCacheError(
                    f"Cache configuration mismatch in {self.cache_file.name}:\n"
                    + "\n".join(f"  - {m}" for m in mismatches)
                )

            # Unpack video entries
            video_ids = data["video_ids"]
            file_paths = data["file_paths"]
            sha256s = data["sha256s"]
            mtimes = data["mtimes"]
            durations = data["durations"]
            timestamps_json = data["timestamps_json"]
            pooled_features = data["pooled_features"]
            per_frame_features = data["per_frame_features"]

            self.records.clear()
            for i, vid in enumerate(video_ids):
                vid_str = str(vid)
                ts_list = json.loads(str(timestamps_json[i]))
                self.records[vid_str] = CachedVideoRecord(
                    video_id=vid_str,
                    file_path=str(file_paths[i]),
                    sha256=str(sha256s[i]),
                    mtime=float(mtimes[i]),
                    duration_sec=float(durations[i]),
                    timestamps_sec=ts_list,
                    pooled_feature=pooled_features[i],
                    per_frame_features=per_frame_features[i],
                )

        except IncompatibleCacheError:
            raise
        except Exception as e:
            raise IncompatibleCacheError(f"Failed to read corrupted cache file {self.cache_file}: {e}")

    def save(self) -> None:
        """
        Save cache to disk using atomic write (writes to temporary file, then renames).
        Prevents corrupting the cache if interrupted.
        """
        if not self.records:
            return

        video_ids = []
        file_paths = []
        sha256s = []
        mtimes = []
        durations = []
        timestamps_json = []
        pooled_features = []
        per_frame_features = []

        for vid, r in self.records.items():
            video_ids.append(r.video_id)
            file_paths.append(r.file_path)
            sha256s.append(r.sha256)
            mtimes.append(r.mtime)
            durations.append(r.duration_sec)
            timestamps_json.append(json.dumps(r.timestamps_sec))
            pooled_features.append(r.pooled_feature)
            per_frame_features.append(r.per_frame_features)

        temp_file = self.cache_file.with_suffix(".tmp.npz")

        np.savez_compressed(
            temp_file,
            header_json=json.dumps(self.metadata_header),
            video_ids=np.array(video_ids),
            file_paths=np.array(file_paths),
            sha256s=np.array(sha256s),
            mtimes=np.array(mtimes, dtype=np.float64),
            durations=np.array(durations, dtype=np.float32),
            timestamps_json=np.array(timestamps_json),
            pooled_features=np.array(pooled_features, dtype=np.float32),
            per_frame_features=np.array(per_frame_features, dtype=np.float32),
        )

        # Atomic replacement on Windows
        if self.cache_file.exists():
            os.replace(temp_file, self.cache_file)
        else:
            temp_file.rename(self.cache_file)

    def is_cached_and_current(self, video_id: str, file_path: Path, sha256: str) -> bool:
        """Check if video feature is present in cache and has not been modified."""
        if video_id not in self.records:
            return False

        rec = self.records[video_id]
        if rec.sha256 != sha256:
            return False

        # Secondary check: filesystem mtime
        if file_path.exists():
            current_mtime = file_path.stat().st_mtime
            if abs(rec.mtime - current_mtime) > 1.0:
                return False

        return True

    def process_and_cache_video(
        self,
        video_id: str,
        video_path: Path,
        sha256: str,
        force_recompute: bool = False,
    ) -> Tuple[CachedVideoRecord, bool]:
        """
        Process a single video: extracts features if new or modified, otherwise reuses cache.
        Returns: (CachedVideoRecord, was_recomputed: bool)
        """
        path_obj = Path(video_path).resolve()
        if not force_recompute and self.is_cached_and_current(video_id, path_obj, sha256):
            return self.records[video_id], False

        # Decode, sample, and preprocess
        clip = self.preprocessor.decode_and_sample(path_obj)
        # Extract features
        per_frame, pooled = self.extractor.extract(clip.tensor)

        mtime = path_obj.stat().st_mtime if path_obj.exists() else 0.0

        record = CachedVideoRecord(
            video_id=video_id,
            file_path=str(path_obj),
            sha256=sha256,
            mtime=mtime,
            duration_sec=clip.duration_sec,
            timestamps_sec=clip.timestamps_sec,
            pooled_feature=pooled,
            per_frame_features=per_frame,
        )

        self.records[video_id] = record
        return record, True
