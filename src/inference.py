"""
Production-Oriented CPU Inference, Temporal Evidence, and Risk Assessment Engine.
Guarding the Stones — Egocentric Heritage Protection AI.

Key Capabilities:
1. End-to-end CPU video inference pipeline (OpenCV decode -> Preprocessing -> Feature Extractor -> Temporal Action Model).
2. Action classification with confidence scores and continuous domain risk scoring.
3. Temporal evidence alignment: preserves timestamps and identifies salient sampled frames via attentive pooling.
4. Configurable review and alert thresholds with explicit uncertainty states.
5. Tri-tier action categorization: Benign Activity, Ambiguous Interactions, and Potential Damage Actions.
6. Explicit ethical & operational guardrails: does not claim confirmed physical damage or fine-grained boundary localization.
7. Robust error handling for corrupt, empty, unreadable files and model mismatches.
8. JSON and CSV report export.

Mathematical Distinction: Confidence vs. Real-World Risk:
- Classification Confidence: P(y_hat | X) = max_c P(y=c | X) measures internal statistical certainty
  over the training taxonomy. A visitor pointing at a carving may have 99% confidence but 0.0 risk.
- Heritage Risk Score: Risk(X) = sum_c [ P(y=c | X) * W_c ], where W_c in [0, 1] represents domain
  physical alteration severity. A 35% probability of 'potential_chipping' carries high operational
  risk (R >= 0.35), warranting immediate human review even if 'inspection' (65%) is the argmax class.
"""
import csv
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json
import os
from pathlib import Path
import time
from typing import Dict, Any, List, Optional, Tuple, Union, Callable

import numpy as np
import torch
import torch.nn.functional as F

from src.config import (
    CANDIDATE_ACTION_CLASSES,
    SUPPORTED_VIDEO_EXTENSIONS,
)
from src.preprocessing import VideoPreprocessor, SampledClip, VideoDecodingError
from src.feature_extractor import FeatureExtractor, BACKBONE_FEATURE_DIMS
from src.models import (
    ModelConfig,
    BaseTemporalActionModel,
    TemporalConv1DNet,
    TemporalPoolingBaseline,
    TemporalGRUNet,
    SklearnBaselineWrapper,
    build_model,
    DEFAULT_ACTION_RISK_WEIGHTS,
)


MODEL_ENGINE_VERSION = "1.0.0-cpu"

# Tri-tier action categories for heritage protection
BENIGN_ACTION_CLASSES = frozenset([
    "normal_looking",
    "walking",
    "pointing",
    "photographing",
    "permitted_touching",
    "normal",
])

AMBIGUOUS_ACTION_CLASSES = frozenset([
    "inspection",
    "ambiguous_interaction",
    "unverified",
])

POTENTIAL_DAMAGE_CLASSES = frozenset([
    "potential_scratching",
    "potential_scraping",
    "potential_chipping",
    "potential_marking",
    "carving",
])


class ModelInferenceError(Exception):
    """Raised when model loading, tensor feeding, or prediction fails."""
    pass


@dataclass
class SampledFrameEvidence:
    """Temporal evidence for a sampled frame supporting the prediction."""
    sample_index: int               # 0-based index in the sampled sequence (0..T-1)
    frame_index: int                # Original video frame index
    timestamp_sec: float            # Exact timestamp in seconds
    attention_weight: float         # Saliency / attention weight assigned by temporal model
    is_peak_evidence: bool = False  # True for the frame with maximum attention

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PredictionResult:
    """Complete prediction report for a single video clip."""
    video_path: str
    status: str                                     # 'success' or 'error'
    predicted_label: str
    confidence: float                               # Max class probability [0.0, 1.0]
    risk_score: float                               # Continuous domain risk [0.0, 1.0]
    action_category: str                            # 'benign_activity', 'ambiguous_interaction', 'potential_damage'
    verdict: str                                    # 'NORMAL', 'UNCERTAIN / HUMAN REVIEW', 'POTENTIAL VANDALISM', 'ERROR'
    uncertainty_flag: bool                          # True if low confidence, high entropy, or ambiguous
    probabilities: Dict[str, float]                 # Full class probability distribution
    peak_evidence_timestamp_sec: Optional[float]   # Timestamp of frame with highest temporal saliency
    peak_evidence_frame_idx: Optional[int]          # Frame index of peak evidence
    sampled_frames: List[SampledFrameEvidence]      # Frame-by-frame temporal evidence
    video_duration_sec: float                       # Total clip duration in seconds
    video_fps: float                                # Video frame rate
    video_total_frames: int                         # Total available frames in clip
    decode_and_preprocess_ms: float                 # Time taken for OpenCV decode + resize + normalize
    inference_ms: float                             # Time taken for feature extraction + temporal model
    total_latency_ms: float                         # Total end-to-end CPU latency
    model_version: str = MODEL_ENGINE_VERSION
    model_type: str = "conv1d"
    backbone: str = "mobilenet_v3_small"
    num_frames: int = 8
    localization_type: str = "clip_level_with_temporal_frame_saliency"
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    disclaimer: str = (
        "Action recognition detects temporal movement patterns. It does NOT confirm physical damage "
        "or malicious intent. Human review by heritage staff is mandatory before taking enforcement action."
    )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["sampled_frames"] = [f.to_dict() for f in self.sampled_frames]
        return d


@dataclass
class BatchPredictionReport:
    """Consolidated report for a batch of video inferences."""
    timestamp: str
    model_version: str
    model_type: str
    backbone: str
    num_frames: int
    review_threshold: float
    alert_threshold: float
    confidence_threshold: float
    total_videos: int
    successful: int
    errors: int
    verdict_counts: Dict[str, int]
    category_counts: Dict[str, int]
    average_latency_ms: float
    total_processing_sec: float
    results: List[PredictionResult]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["results"] = [r.to_dict() for r in self.results]
        return d


def categorize_action_label(label: str) -> str:
    """Categorize an action label into benign, ambiguous, or potential damage."""
    lbl = label.lower().strip()
    if lbl in BENIGN_ACTION_CLASSES:
        return "benign_activity"
    elif lbl in POTENTIAL_DAMAGE_CLASSES:
        return "potential_damage"
    elif lbl in AMBIGUOUS_ACTION_CLASSES:
        return "ambiguous_interaction"
    return "ambiguous_interaction"


def compute_verdict(
    predicted_label: str,
    confidence: float,
    risk_score: float,
    review_threshold: float = 0.35,
    alert_threshold: float = 0.70,
    confidence_threshold: float = 0.40,
) -> Tuple[str, bool]:
    """
    Compute operational security verdict and uncertainty flag based on risk and confidence.

    Rules:
    1. If confidence < confidence_threshold -> UNCERTAIN / HUMAN REVIEW (uncertainty_flag=True)
    2. If risk_score >= alert_threshold and action is potential damage -> POTENTIAL VANDALISM
    3. If risk_score >= review_threshold or action is ambiguous -> UNCERTAIN / HUMAN REVIEW
    4. If risk_score < review_threshold and action is benign -> NORMAL
    5. Fallback -> UNCERTAIN / HUMAN REVIEW
    """
    category = categorize_action_label(predicted_label)
    uncertainty_flag = False

    # Check for low confidence uncertainty
    if confidence < confidence_threshold:
        uncertainty_flag = True
        return "UNCERTAIN / HUMAN REVIEW", uncertainty_flag

    # Check for potential vandalism alert
    if risk_score >= alert_threshold and category == "potential_damage":
        return "POTENTIAL VANDALISM", False

    # Check for human review boundary
    if risk_score >= review_threshold or category in ("ambiguous_interaction", "potential_damage"):
        return "UNCERTAIN / HUMAN REVIEW", True

    # Benign action with low risk
    if category == "benign_activity" and risk_score < review_threshold:
        return "NORMAL", False

    return "UNCERTAIN / HUMAN REVIEW", True


class InferenceEngine:
    """
    Production-grade CPU Video Inference Engine for Egocentric Heritage Action Recognition.
    Coordinates video decoding, frame sampling, feature extraction, temporal classification,
    continuous risk calculation, and temporal evidence aggregation.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        model: Optional[Any] = None,
        config: Optional[ModelConfig] = None,
        label_to_idx: Optional[Dict[str, int]] = None,
        review_threshold: float = 0.35,
        alert_threshold: float = 0.70,
        confidence_threshold: float = 0.40,
        cpu_threads: Optional[int] = None,
    ):
        self.review_threshold = float(review_threshold)
        self.alert_threshold = float(alert_threshold)
        self.confidence_threshold = float(confidence_threshold)
        self.cpu_threads = cpu_threads or os.cpu_count() or 4
        torch.set_num_threads(self.cpu_threads)

        # 1. Resolve Model and Configuration
        self.model, self.config, self.label_to_idx = self._initialize_model(
            model_path=model_path,
            model=model,
            config=config,
            label_to_idx=label_to_idx,
        )

        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}
        self.class_names = [self.idx_to_label[i] for i in range(len(self.idx_to_label))]

        # 2. Build Preprocessor and Feature Extractor
        self.preprocessor = VideoPreprocessor(
            num_frames=self.config.num_frames,
            img_size=224,
        )
        self.feature_extractor = FeatureExtractor(
            backbone_name=self.config.backbone,
            weights="none",
            allow_download=False,
            cpu_threads=self.cpu_threads,
        )

    def _initialize_model(
        self,
        model_path: Optional[Union[str, Path]],
        model: Optional[Any],
        config: Optional[ModelConfig],
        label_to_idx: Optional[Dict[str, int]],
    ) -> Tuple[Any, ModelConfig, Dict[str, int]]:
        """Load model weights or construct model instance safely."""
        # Case A: Model instance provided directly
        if model is not None:
            if isinstance(model, BaseTemporalActionModel):
                m_config = model.config
                m_label_map = label_to_idx or {name: i for i, name in enumerate(m_config.class_names)}
                model.eval()
                return model, m_config, m_label_map
            elif isinstance(model, SklearnBaselineWrapper):
                m_config = config or ModelConfig(model_type="pooling_mlp")
                m_label_map = label_to_idx or {c: i for i, c in enumerate(m_config.class_names)}
                return model, m_config, m_label_map
            else:
                raise TypeError(f"Unsupported model instance type: {type(model)}")

        # Case B: Load from checkpoint path
        if model_path is not None:
            p = Path(model_path).resolve()
            if not p.exists():
                raise FileNotFoundError(f"Model checkpoint does not exist: {p}")

            if p.suffix.lower() in (".pt", ".pth"):
                checkpoint = torch.load(p, map_location="cpu", weights_only=False)
                if isinstance(checkpoint, dict) and "model_config" in checkpoint:
                    m_config = ModelConfig.from_dict(checkpoint["model_config"])
                    m_label_map = checkpoint.get("label_to_idx", {name: i for i, name in enumerate(m_config.class_names)})
                    loaded_model = build_model(m_config)
                    if "model_state_dict" in checkpoint:
                        loaded_model.load_state_dict(checkpoint["model_state_dict"])
                    loaded_model.eval()
                    return loaded_model, m_config, m_label_map
                elif isinstance(checkpoint, BaseTemporalActionModel):
                    checkpoint.eval()
                    m_config = checkpoint.config
                    m_label_map = label_to_idx or {name: i for i, name in enumerate(m_config.class_names)}
                    return checkpoint, m_config, m_label_map
                else:
                    raise ModelInferenceError(f"Unrecognized PyTorch checkpoint format at {p}")
            elif p.suffix.lower() in (".joblib", ".pkl"):
                import joblib
                loaded_model = joblib.load(p)
                m_config = config or ModelConfig(model_type="pooling_mlp")
                m_label_map = label_to_idx or {c: i for i, c in enumerate(m_config.class_names)}
                return loaded_model, m_config, m_label_map
            else:
                raise ValueError(f"Unsupported model file extension: {p.suffix}")

        # Case C: Construct default model from config or defaults
        default_config = config or ModelConfig(
            model_type="conv1d",
            backbone="mobilenet_v3_small",
            num_frames=8,
            class_names=sorted(list(CANDIDATE_ACTION_CLASSES)),
        )
        built_model = build_model(default_config)
        built_model.eval()
        default_label_map = label_to_idx or {name: i for i, name in enumerate(default_config.class_names)}
        return built_model, default_config, default_label_map

    def predict_video(self, video_path: Union[str, Path]) -> PredictionResult:
        """
        Run complete end-to-end inference on a single video file.
        Catches decoding errors and returns structured error result without throwing.
        """
        p = Path(video_path).resolve()
        path_str = str(p)

        # Basic file validation
        if not p.exists():
            return self._make_error_result(path_str, "FILE_NOT_FOUND", f"File does not exist: {p}")

        if p.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
            return self._make_error_result(
                path_str,
                "UNSUPPORTED_FORMAT",
                f"Unsupported extension '{p.suffix}'. Supported: {sorted(list(SUPPORTED_VIDEO_EXTENSIONS))}",
            )

        if p.stat().st_size == 0:
            return self._make_error_result(path_str, "EMPTY_FILE", "Video file size is 0 bytes.")

        t_total_start = time.perf_counter()

        # Step 1: Decode and sample frames
        t_decode_start = time.perf_counter()
        try:
            sampled: SampledClip = self.preprocessor.decode_and_sample(p)
        except (VideoDecodingError, Exception) as e:
            return self._make_error_result(path_str, "DECODING_ERROR", f"Failed to decode video: {str(e)}")

        decode_and_preprocess_ms = (time.perf_counter() - t_decode_start) * 1000.0

        # Step 2: Feature Extraction and Temporal Inference
        t_infer_start = time.perf_counter()
        try:
            per_frame_feats, pooled_feat = self.feature_extractor.extract(sampled.tensor)
        except Exception as e:
            return self._make_error_result(path_str, "FEATURE_EXTRACTION_ERROR", f"Feature extractor failed: {str(e)}")

        # Step 3: Action Classification Model Execution
        attention_weights: Optional[np.ndarray] = None

        if isinstance(self.model, torch.nn.Module):
            with torch.no_grad():
                # Shape: (1, T, D)
                x = torch.tensor(per_frame_feats, dtype=torch.float32).unsqueeze(0)
                try:
                    out = self.model(x)
                except Exception as e:
                    return self._make_error_result(path_str, "MODEL_EXECUTION_ERROR", f"Model forward pass failed: {str(e)}")

                probs = out["probabilities"].cpu().numpy()[0]
                risk = float(out["risk_score"].cpu().numpy()[0])
                if "attention_weights" in out:
                    attention_weights = out["attention_weights"].cpu().numpy()[0]
        elif isinstance(self.model, SklearnBaselineWrapper):
            try:
                probs = self.model.predict_proba(pooled_feat.reshape(1, -1))[0]
            except Exception as e:
                return self._make_error_result(path_str, "MODEL_EXECUTION_ERROR", f"Baseline prediction failed: {str(e)}")

            # Compute continuous risk
            risk_vector = np.array([self.config.risk_weights.get(c, 0.0) for c in self.class_names], dtype=np.float32)
            if len(probs) == len(risk_vector):
                risk = float(np.dot(probs, risk_vector))
            else:
                risk = float(probs[-1])  # Fallback
        else:
            return self._make_error_result(path_str, "INVALID_MODEL_TYPE", f"Unknown model type: {type(self.model)}")

        inference_ms = (time.perf_counter() - t_infer_start) * 1000.0
        total_latency_ms = (time.perf_counter() - t_total_start) * 1000.0

        # Step 4: Map probabilities and select top predicted class
        prob_dict: Dict[str, float] = {}
        for idx, p_val in enumerate(probs):
            if idx < len(self.class_names):
                prob_dict[self.class_names[idx]] = float(round(p_val, 4))

        pred_idx = int(np.argmax(probs))
        predicted_label = self.class_names[pred_idx] if pred_idx < len(self.class_names) else "unverified"
        confidence = float(round(probs[pred_idx], 4))
        risk_score = float(round(np.clip(risk, 0.0, 1.0), 4))

        # Step 5: Align temporal frame evidence and identify peak evidence frame
        num_sampled = len(sampled.timestamps_sec)
        if attention_weights is None or len(attention_weights) != num_sampled:
            # Uniform attention baseline if model does not output attention weights
            attention_weights = np.ones(num_sampled, dtype=np.float32) / max(num_sampled, 1)

        peak_idx_local = int(np.argmax(attention_weights))
        peak_timestamp = float(round(sampled.timestamps_sec[peak_idx_local], 3))
        peak_frame_idx = int(sampled.frame_indices[peak_idx_local])

        frame_evidence_list: List[SampledFrameEvidence] = []
        for i in range(num_sampled):
            frame_evidence_list.append(
                SampledFrameEvidence(
                    sample_index=i,
                    frame_index=int(sampled.frame_indices[i]),
                    timestamp_sec=float(round(sampled.timestamps_sec[i], 3)),
                    attention_weight=float(round(attention_weights[i], 4)),
                    is_peak_evidence=(i == peak_idx_local),
                )
            )

        # Step 6: Determine Action Category, Operational Verdict, and Uncertainty Flag
        action_cat = categorize_action_label(predicted_label)
        verdict, uncertainty = compute_verdict(
            predicted_label=predicted_label,
            confidence=confidence,
            risk_score=risk_score,
            review_threshold=self.review_threshold,
            alert_threshold=self.alert_threshold,
            confidence_threshold=self.confidence_threshold,
        )

        return PredictionResult(
            video_path=path_str,
            status="success",
            predicted_label=predicted_label,
            confidence=confidence,
            risk_score=risk_score,
            action_category=action_cat,
            verdict=verdict,
            uncertainty_flag=uncertainty,
            probabilities=prob_dict,
            peak_evidence_timestamp_sec=peak_timestamp,
            peak_evidence_frame_idx=peak_frame_idx,
            sampled_frames=frame_evidence_list,
            video_duration_sec=float(round(sampled.duration_sec, 2)),
            video_fps=float(round(sampled.fps, 2)),
            video_total_frames=int(sampled.total_frames_in_video),
            decode_and_preprocess_ms=float(round(decode_and_preprocess_ms, 2)),
            inference_ms=float(round(inference_ms, 2)),
            total_latency_ms=float(round(total_latency_ms, 2)),
            model_version=MODEL_ENGINE_VERSION,
            model_type=self.config.model_type,
            backbone=self.config.backbone,
            num_frames=self.config.num_frames,
            localization_type="clip_level_with_temporal_frame_saliency",
        )

    def predict_video_with_frames(
        self, video_path: Union[str, Path]
    ) -> Tuple[PredictionResult, Optional[List[Tuple[np.ndarray, str]]]]:
        """
        Run end-to-end inference and return both the structured PredictionResult
        and the sampled RGB frames with captions suitable for UI gallery visualization.
        """
        p = Path(video_path).resolve()
        path_str = str(p)

        if not p.exists():
            return self._make_error_result(path_str, "FILE_NOT_FOUND", f"File does not exist: {p}"), None

        try:
            sampled: SampledClip = self.preprocessor.decode_and_sample(p)
        except (VideoDecodingError, Exception) as e:
            return self._make_error_result(path_str, "DECODING_ERROR", f"Failed to decode video: {str(e)}"), None

        # Run core prediction
        res = self.predict_video(p)
        if res.status != "success":
            return res, None

        # Build gallery pairs (frame_rgb, caption)
        gallery_items: List[Tuple[np.ndarray, str]] = []
        for i, f_ev in enumerate(res.sampled_frames):
            frame_rgb = sampled.frames_rgb[i]
            peak_tag = " ★ [PEAK EVIDENCE]" if f_ev.is_peak_evidence else ""
            caption = (
                f"Sample {i + 1}/{len(res.sampled_frames)} | Frame {f_ev.frame_index} | "
                f"{f_ev.timestamp_sec:.2f}s | Attn: {f_ev.attention_weight:.3f}{peak_tag}"
            )
            gallery_items.append((frame_rgb, caption))

        return res, gallery_items

    def _make_error_result(self, video_path: str, error_type: str, message: str) -> PredictionResult:
        """Construct graceful error result when video processing fails."""
        return PredictionResult(
            video_path=video_path,
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
            model_version=MODEL_ENGINE_VERSION,
            model_type=self.config.model_type,
            backbone=self.config.backbone,
            num_frames=self.config.num_frames,
            localization_type="clip_level_with_temporal_frame_saliency",
            error_type=error_type,
            error_message=message,
        )

    def predict_batch(
        self,
        video_paths: List[Union[str, Path]],
        progress_callback: Optional[Callable[[int, int, PredictionResult], None]] = None,
    ) -> BatchPredictionReport:
        """
        Run inference over a list of video files sequentially on CPU.
        Never crashes the entire batch if individual videos fail.
        """
        results: List[PredictionResult] = []
        t0 = time.time()
        total = len(video_paths)

        verdict_counts: Dict[str, int] = {
            "NORMAL": 0,
            "UNCERTAIN / HUMAN REVIEW": 0,
            "POTENTIAL VANDALISM": 0,
            "ERROR": 0,
        }
        category_counts: Dict[str, int] = {
            "benign_activity": 0,
            "ambiguous_interaction": 0,
            "potential_damage": 0,
            "unknown": 0,
        }
        latencies: List[float] = []

        for idx, v_path in enumerate(video_paths):
            res = self.predict_video(v_path)
            results.append(res)

            verdict_counts[res.verdict] = verdict_counts.get(res.verdict, 0) + 1
            category_counts[res.action_category] = category_counts.get(res.action_category, 0) + 1

            if res.status == "success":
                latencies.append(res.total_latency_ms)

            if progress_callback is not None:
                progress_callback(idx + 1, total, res)

        total_sec = time.time() - t0
        avg_lat = float(round(np.mean(latencies), 2)) if latencies else 0.0
        success_count = sum(1 for r in results if r.status == "success")
        error_count = total - success_count

        return BatchPredictionReport(
            timestamp=datetime.now().isoformat(),
            model_version=MODEL_ENGINE_VERSION,
            model_type=self.config.model_type,
            backbone=self.config.backbone,
            num_frames=self.config.num_frames,
            review_threshold=self.review_threshold,
            alert_threshold=self.alert_threshold,
            confidence_threshold=self.confidence_threshold,
            total_videos=total,
            successful=success_count,
            errors=error_count,
            verdict_counts=verdict_counts,
            category_counts=category_counts,
            average_latency_ms=avg_lat,
            total_processing_sec=float(round(total_sec, 2)),
            results=results,
        )


# =====================================================================
# Export Functions (JSON & CSV)
# =====================================================================

def export_predictions_json(report: Union[BatchPredictionReport, PredictionResult], out_path: Union[str, Path]) -> Path:
    """Export predictions report to structured JSON format."""
    p = Path(out_path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)
    return p


def export_predictions_csv(report: BatchPredictionReport, out_path: Union[str, Path]) -> Path:
    """Export batch predictions to flattened CSV format."""
    p = Path(out_path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "video_path",
        "status",
        "predicted_label",
        "confidence",
        "risk_score",
        "action_category",
        "verdict",
        "uncertainty_flag",
        "peak_evidence_timestamp_sec",
        "peak_evidence_frame_idx",
        "video_duration_sec",
        "video_fps",
        "decode_preprocess_ms",
        "inference_ms",
        "total_latency_ms",
        "error_type",
        "error_message",
    ]

    with open(p, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in report.results:
            writer.writerow({
                "video_path": r.video_path,
                "status": r.status,
                "predicted_label": r.predicted_label,
                "confidence": r.confidence,
                "risk_score": r.risk_score,
                "action_category": r.action_category,
                "verdict": r.verdict,
                "uncertainty_flag": r.uncertainty_flag,
                "peak_evidence_timestamp_sec": r.peak_evidence_timestamp_sec,
                "peak_evidence_frame_idx": r.peak_evidence_frame_idx,
                "video_duration_sec": r.video_duration_sec,
                "video_fps": r.video_fps,
                "decode_preprocess_ms": r.decode_and_preprocess_ms,
                "inference_ms": r.inference_ms,
                "total_latency_ms": r.total_latency_ms,
                "error_type": r.error_type or "",
                "error_message": r.error_message or "",
            })
    return p


def save_inference_package(
    model: BaseTemporalActionModel,
    out_path: Union[str, Path],
    label_to_idx: Optional[Dict[str, int]] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save complete self-contained deployment package for CPU inference."""
    p = Path(out_path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)

    package = {
        "model_version": MODEL_ENGINE_VERSION,
        "export_timestamp": datetime.now().isoformat(),
        "model_config": model.config.to_dict(),
        "model_state_dict": model.state_dict(),
        "label_to_idx": label_to_idx or {name: i for i, name in enumerate(model.config.class_names)},
        "metadata": extra_metadata or {},
    }
    torch.save(package, p)
    return p
