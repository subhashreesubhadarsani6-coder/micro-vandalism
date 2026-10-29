"""
Comprehensive Evaluation and Benchmarking Engine.
Guarding the Stones — Egocentric Heritage Protection AI.
Measures Accuracy, Balanced Accuracy, Macro/Weighted F1, ROC/PR-AUC,
Expected Calibration Error (ECE), Brier Score, and CPU Latency/Throughput/Memory.
"""
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json
import os
from pathlib import Path
import time
import tracemalloc
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    auc,
    brier_score_loss,
)

from src.models import (
    ModelConfig,
    BaseTemporalActionModel,
    TemporalConv1DNet,
    TemporalPoolingBaseline,
    TemporalGRUNet,
    SklearnBaselineWrapper,
)


@dataclass
class EvaluationMetrics:
    """Consolidated metrics container for model evaluation and benchmarking."""
    model_name: str
    backbone: str
    num_frames: int
    dataset_version: str
    num_test_samples: int
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    weighted_f1: float
    macro_precision: float
    macro_recall: float
    roc_auc: Optional[float]
    pr_auc: Optional[float]
    brier_score: Optional[float]
    expected_calibration_error: Optional[float]
    confusion_matrix: List[List[int]]
    class_support: Dict[str, int]
    per_class_metrics: Dict[str, Dict[str, float]]
    latency_mean_ms: float
    latency_median_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    throughput_clips_sec: float
    effective_fps: float
    peak_memory_mb: float
    missing_classes: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_expected_calibration_error(
    probabilities: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE):
    Partitions confidence predictions into M bins and calculates weighted difference
    between accuracy and confidence across all bins.
    """
    if len(targets) == 0:
        return 0.0

    if probabilities.ndim == 2:
        confidences = np.max(probabilities, axis=1)
        predictions = np.argmax(probabilities, axis=1)
    else:
        confidences = probabilities
        predictions = (probabilities >= 0.5).astype(int)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_total = len(targets)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Samples within bin range
        if i == n_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(predictions[in_bin] == targets[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (bin_size / n_total) * np.abs(bin_acc - bin_conf)

    return float(round(ece, 4))


def evaluate_model_on_test_set(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str],
    model_name: str = "TemporalConv1DNet",
    backbone: str = "mobilenet_v3_small",
    num_frames: int = 8,
    dataset_version: str = "v1.0-proxy",
    limitations: Optional[List[str]] = None,
) -> EvaluationMetrics:
    """
    Run complete CPU evaluation and latency benchmarking on held-out test data.
    Never tunes or retrains model on test data.
    """
    n_samples = len(y_test)
    if n_samples == 0:
        raise ValueError("Cannot evaluate on empty test set.")

    # Latency and memory tracing on CPU
    tracemalloc.start()
    latencies_ms: List[float] = []

    is_torch_model = isinstance(model, torch.nn.Module)
    if is_torch_model:
        model.eval()

    y_probs_list = []
    y_preds_list = []

    # Single-sample inference loop to measure true per-video latency
    t_start_total = time.perf_counter()
    with torch.no_grad():
        for i in range(n_samples):
            x_sample = X_test[i : i + 1]  # Shape: (1, T, D) or (1, 2D)
            t0 = time.perf_counter()

            if is_torch_model:
                tensor_in = torch.tensor(x_sample, dtype=torch.float32)
                out = model(tensor_in)
                probs = out["probabilities"].cpu().numpy()[0]
                pred = int(np.argmax(probs))
            else:
                # Scikit-learn wrapper
                probs = model.predict_proba(x_sample)[0]
                pred = int(np.argmax(probs))

            t_lat = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(t_lat)
            y_probs_list.append(probs)
            y_preds_list.append(pred)

    total_eval_time = time.perf_counter() - t_start_total
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    y_probs = np.array(y_probs_list)
    y_preds = np.array(y_preds_list)

    # 1. Classification Metrics
    acc = float(accuracy_score(y_test, y_preds))
    bal_acc = float(balanced_accuracy_score(y_test, y_preds))
    macro_f1 = float(f1_score(y_test, y_preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_test, y_preds, average="weighted", zero_division=0))
    macro_prec = float(precision_score(y_test, y_preds, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_test, y_preds, average="macro", zero_division=0))

    # Confusion matrix
    conf_mat = confusion_matrix(y_test, y_preds).tolist()

    # Per-class metrics
    present_classes = sorted(list(set(np.concatenate([y_test, y_preds]))))
    per_class: Dict[str, Dict[str, float]] = {}
    class_support: Dict[str, int] = {}
    missing_classes: List[str] = []

    for idx, c_name in enumerate(class_names):
        count = int(np.sum(y_test == idx))
        class_support[c_name] = count
        if count == 0:
            missing_classes.append(c_name)
        else:
            p_c = float(precision_score(y_test == idx, y_preds == idx, zero_division=0))
            r_c = float(recall_score(y_test == idx, y_preds == idx, zero_division=0))
            f1_c = float(f1_score(y_test == idx, y_preds == idx, zero_division=0))
            per_class[c_name] = {
                "precision": round(p_c, 4),
                "recall": round(r_c, 4),
                "f1": round(f1_c, 4),
                "support": count,
            }

    # 2. Probability & Calibration Metrics
    roc_auc = None
    pr_auc = None
    brier = None

    unique_targets = np.unique(y_test)
    if len(unique_targets) == 2:
        # Binary classification metrics
        pos_idx = 1 if y_probs.shape[1] > 1 else 0
        pos_probs = y_probs[:, pos_idx]
        try:
            roc_auc = float(round(roc_auc_score(y_test, pos_probs), 4))
        except Exception:
            pass
        try:
            p_curve, r_curve, _ = precision_recall_curve(y_test, pos_probs)
            pr_auc = float(round(auc(r_curve, p_curve), 4))
        except Exception:
            pass
        try:
            brier = float(round(brier_score_loss(y_test, pos_probs), 4))
        except Exception:
            pass
    elif len(unique_targets) > 2 and y_probs.shape[1] == len(class_names):
        # Multiclass ROC-AUC (OVR)
        try:
            roc_auc = float(round(roc_auc_score(y_test, y_probs, multi_class="ovr", average="macro", labels=unique_targets), 4))
        except Exception:
            pass

    ece = compute_expected_calibration_error(y_probs, y_test, n_bins=10)

    # 3. CPU Benchmarking Stats
    lat_mean = float(round(np.mean(latencies_ms), 2))
    lat_median = float(round(np.median(latencies_ms), 2))
    lat_p95 = float(round(np.percentile(latencies_ms, 95), 2))
    lat_p99 = float(round(np.percentile(latencies_ms, 99), 2))
    clips_sec = float(round(n_samples / total_eval_time, 2)) if total_eval_time > 0 else 0.0
    effective_fps = float(round(clips_sec * num_frames, 2))
    peak_mem_mb = float(round(peak_bytes / (1024 * 1024), 2))

    return EvaluationMetrics(
        model_name=model_name,
        backbone=backbone,
        num_frames=num_frames,
        dataset_version=dataset_version,
        num_test_samples=n_samples,
        accuracy=round(acc, 4),
        balanced_accuracy=round(bal_acc, 4),
        macro_f1=round(macro_f1, 4),
        weighted_f1=round(weighted_f1, 4),
        macro_precision=round(macro_prec, 4),
        macro_recall=round(macro_rec, 4),
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        brier_score=brier,
        expected_calibration_error=ece,
        confusion_matrix=conf_mat,
        class_support=class_support,
        per_class_metrics=per_class,
        latency_mean_ms=lat_mean,
        latency_median_ms=lat_median,
        latency_p95_ms=lat_p95,
        latency_p99_ms=lat_p99,
        throughput_clips_sec=clips_sec,
        effective_fps=effective_fps,
        peak_memory_mb=peak_mem_mb,
        missing_classes=missing_classes,
        limitations=limitations or [],
    )


def export_evaluation_results(
    metrics_list: List[EvaluationMetrics],
    out_dir: Path,
) -> Dict[str, str]:
    """Export evaluation metrics to JSON and CSV formats."""
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "evaluation_metrics.json"
    csv_summary_path = out_dir / "evaluation_summary.csv"
    csv_per_class_path = out_dir / "evaluation_per_class.csv"

    # Export full JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([m.to_dict() for m in metrics_list], f, indent=2)

    # Export summary CSV
    import csv
    summary_headers = [
        "model_name", "backbone", "num_frames", "num_test_samples",
        "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1",
        "macro_precision", "macro_recall", "roc_auc", "pr_auc",
        "brier_score", "expected_calibration_error",
        "latency_mean_ms", "latency_p95_ms", "throughput_clips_sec", "peak_memory_mb"
    ]
    with open(csv_summary_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_headers)
        writer.writeheader()
        for m in metrics_list:
            d = m.to_dict()
            row = {h: d.get(h) for h in summary_headers}
            writer.writerow(row)

    # Export per-class CSV
    per_class_headers = ["model_name", "num_frames", "class_name", "precision", "recall", "f1", "support"]
    with open(csv_per_class_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=per_class_headers)
        writer.writeheader()
        for m in metrics_list:
            for c_name, c_data in m.per_class_metrics.items():
                writer.writerow({
                    "model_name": m.model_name,
                    "num_frames": m.num_frames,
                    "class_name": c_name,
                    "precision": c_data["precision"],
                    "recall": c_data["recall"],
                    "f1": c_data["f1"],
                    "support": c_data["support"],
                })

    return {
        "json": str(json_path),
        "csv_summary": str(csv_summary_path),
        "csv_per_class": str(csv_per_class_path),
    }
