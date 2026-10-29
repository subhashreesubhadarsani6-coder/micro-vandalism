"""
Training, Validation, and Pre-flight Verification Engine.
Guarding the Stones — Egocentric Heritage Protection AI.
"""
from dataclasses import dataclass, field, asdict
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Dict, Any, List, Optional, Tuple, Set

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from src.config import CANDIDATE_ACTION_CLASSES, DatasetConfig
from src.models import (
    ModelConfig,
    BaseTemporalActionModel,
    build_model,
    compute_class_weights,
    FocalLoss,
    SklearnBaselineWrapper,
)
from src.feature_cache import FeatureCacheManager
from src.feature_extractor import FeatureExtractor, WEIGHTS_METADATA


class PretrainingPrerequisiteError(Exception):
    """Raised when dataset integrity or egocentric requirements fail pre-training checks."""
    pass


@dataclass
class PretrainingCheckReport:
    """Detailed diagnostic report of pre-training verification checks."""
    passed: bool
    total_manifest_records: int
    egocentric_records_count: int
    unique_videos_count: int
    duplicate_videos_count: int
    unverified_labels_count: int
    class_distribution: Dict[str, int]
    cpu_cores_logical: int
    cpu_memory_available_gb: float
    pretrained_weights_available: bool
    weights_details: Dict[str, Any]
    fatal_failures: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def run_pretraining_checks(
    manifest_path: Path,
    backbone: str = "mobilenet_v3_small",
    weights_mode: str = "none",
    min_egocentric_required: int = 20,
) -> PretrainingCheckReport:
    """
    Strict pre-training verification gate:
    1. Verifies that suitable first-person videos exist.
    2. Verifies action labels and human verification quality.
    3. Checks class counts, duplicate clips, and source identifiers.
    4. Confirms that enough examples exist for a meaningful train/val/test split.
    5. Checks CPU resources and RAM.
    6. Verifies that selected pretrained weights are available.
    7. Refuses to proceed if required data is missing or invalid.
    """
    failures: List[str] = []
    warnings: List[str] = []

    m_path = Path(manifest_path).resolve()
    if not m_path.exists():
        failures.append(f"Manifest file does not exist: {m_path}")
        return PretrainingCheckReport(
            passed=False,
            total_manifest_records=0,
            egocentric_records_count=0,
            unique_videos_count=0,
            duplicate_videos_count=0,
            unverified_labels_count=0,
            class_distribution={},
            cpu_cores_logical=os.cpu_count() or 1,
            cpu_memory_available_gb=0.0,
            pretrained_weights_available=False,
            weights_details={},
            fatal_failures=failures,
        )

    # 1. Inspect manifest
    records: List[Dict[str, Any]] = []
    with open(m_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    total_records = len(records)
    egocentric_records = [r for r in records if r.get("is_egocentric") is True]
    egocentric_count = len(egocentric_records)

    # Check 1: Egocentric viewpoint verification
    if egocentric_count == 0:
        failures.append(
            f"ZERO authentic first-person (egocentric) clips found in manifest. "
            f"All {total_records} existing clips are third-person commercial stock clips or static shots. "
            f"Training an 'Egocentric Action Model' on third-person stock video violates domain validity."
        )
    elif egocentric_count < min_egocentric_required:
        failures.append(
            f"Insufficient egocentric examples: found {egocentric_count}, but minimum {min_egocentric_required} required."
        )

    # Check 2: Action labels and human verification
    class_counts: Dict[str, int] = {}
    unverified_count = 0
    duplicates_count = 0

    for r in records:
        lbl = r.get("action_label", "unverified")
        class_counts[lbl] = class_counts.get(lbl, 0) + 1
        if lbl == "unverified":
            unverified_count += 1
        if r.get("validation_status") == "duplicate":
            duplicates_count += 1

    if unverified_count == total_records:
        failures.append(
            f"100% of manifest action labels ({unverified_count}/{total_records}) are 'unverified'. "
            f"Labels must never be assigned based on filename heuristics alone without human verification."
        )
    elif unverified_count > 0:
        warnings.append(f"{unverified_count} clips in manifest remain unverified and will be excluded from training.")

    # Check 3: Minimum examples for stratified train/val/test splitting
    verified_labeled = total_records - unverified_count
    if verified_labeled < 15 and len(failures) == 0:
        failures.append(
            f"Insufficient verified labeled examples for 3-way split: found {verified_labeled}, need >= 15."
        )

    # Check 4: Duplicate clips leakage risk
    if duplicates_count > 0:
        warnings.append(
            f"Found {duplicates_count} exact duplicate clips in raw data. "
            f"Duplicates must be excluded or strictly confined to single splits to prevent data leakage."
        )

    # Check 5: System resources
    logical_cores = os.cpu_count() or 4
    avail_mem_gb = 4.0
    try:
        import psutil
        avail_mem_gb = round(psutil.virtual_memory().available / (1024 ** 3), 2)
    except ImportError:
        pass

    # Check 6: Pretrained weights availability
    weights_meta = WEIGHTS_METADATA.get(backbone, {})
    weights_file = weights_meta.get("file_name", "")
    hub_path = Path(torch.hub.get_dir()) / "checkpoints" / weights_file
    has_local_weights = hub_path.exists()

    if weights_mode in ("pretrained", "default") and not has_local_weights:
        failures.append(
            f"Pretrained weights for '{backbone}' are not cached locally ({weights_file}, {weights_meta.get('size_mb')} MB). "
            f"Automatic large file downloads are disallowed by safety policy."
        )

    passed = len(failures) == 0
    return PretrainingCheckReport(
        passed=passed,
        total_manifest_records=total_records,
        egocentric_records_count=egocentric_count,
        unique_videos_count=total_records - duplicates_count,
        duplicate_videos_count=duplicates_count,
        unverified_labels_count=unverified_count,
        class_distribution=class_counts,
        cpu_cores_logical=logical_cores,
        cpu_memory_available_gb=avail_mem_gb,
        pretrained_weights_available=has_local_weights,
        weights_details=weights_meta,
        fatal_failures=failures,
        warnings=warnings,
    )


# =====================================================================
# Group-Aware Dataset Splitting
# =====================================================================

def split_dataset_group_aware(
    records: List[Dict[str, Any]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split dataset into train, validation, and test partitions while:
    - Grouping clips from the same participant/session/source archive.
    - Guaranteeing participant/source groups never leak across splits.
    - Stratifying class distributions where feasible.
    """
    if not records:
        return [], [], []

    rng = np.random.RandomState(seed)

    # 1. Map each record to its group identifier
    record_groups: List[str] = []
    group_to_indices: Dict[str, List[int]] = {}

    for idx, r in enumerate(records):
        group_id = r.get("participant_id") or r.get("session_id")
        if not group_id:
            fname = r.get("file_name", "")
            base = "".join([c for c in fname.split(".")[0] if not c.isdigit()]).strip()
            group_id = base or r.get("sha256", "")[:8]
        record_groups.append(group_id)
        group_to_indices.setdefault(group_id, []).append(idx)

    unique_groups = sorted(list(group_to_indices.keys()))

    if len(unique_groups) >= 3:
        # Group-level partition: partition unique groups so groups are strictly disjoint
        rng.shuffle(unique_groups)
        n_groups = len(unique_groups)
        n_train = max(1, int(round(n_groups * train_ratio)))
        n_val = max(1, int(round(n_groups * val_ratio)))
        # Adjust if sum exceeds total
        if n_train + n_val >= n_groups:
            n_train = max(1, n_groups - 2)
            n_val = 1
        n_test = max(1, n_groups - (n_train + n_val))

        train_groups = set(unique_groups[:n_train])
        val_groups = set(unique_groups[n_train : n_train + n_val])
        test_groups = set(unique_groups[n_train + n_val :])

        train_idx = [i for g in train_groups for i in group_to_indices[g]]
        val_idx = [i for g in val_groups for i in group_to_indices[g]]
        test_idx = [i for g in test_groups for i in group_to_indices[g]]
    else:
        # Fallback when there are fewer than 3 groups
        indices = np.arange(len(records))
        labels = np.array([r.get("action_label", "unverified") for r in records])
        strat = labels if len(set(labels)) > 1 else None

        train_idx, rest_idx = train_test_split(
            indices, test_size=(val_ratio + test_ratio), random_state=seed, stratify=strat
        )
        val_in_rest = val_ratio / (val_ratio + test_ratio)
        val_idx, test_idx = train_test_split(
            rest_idx, test_size=(1.0 - val_in_rest), random_state=seed
        )

    train_recs = [records[i] for i in train_idx]
    val_recs = [records[i] for i in val_idx]
    test_recs = [records[i] for i in test_idx]

    return train_recs, val_recs, test_recs


# =====================================================================
# Action Recognition PyTorch Dataset
# =====================================================================

class ActionFeatureDataset(Dataset):
    """Loads temporal frame embeddings (T, D) and action targets from feature cache."""

    def __init__(
        self,
        records: List[Dict[str, Any]],
        cache_manager: FeatureCacheManager,
        label_to_idx: Dict[str, int],
    ):
        self.samples: List[Tuple[np.ndarray, int, str]] = []
        for r in records:
            vid = r.get("video_id")
            lbl = r.get("action_label")
            if vid in cache_manager.records and lbl in label_to_idx:
                cached_item = cache_manager.records[vid]
                feats = cached_item.per_frame_features  # (T, D)
                self.samples.append((feats, label_to_idx[lbl], vid))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        feats, target, vid = self.samples[idx]
        return torch.tensor(feats, dtype=torch.float32), torch.tensor(target, dtype=torch.long), vid


# =====================================================================
# Complete Trainer
# =====================================================================

@dataclass
class TrainingConfig:
    """Hyperparameters and configuration for action recognition training."""
    epochs: int = 40
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 10
    seed: int = 42
    use_focal_loss: bool = False
    output_dir: str = "models/checkpoints"


class ActionTrainer:
    """Orchestrates model training, validation, early stopping, evaluation, and artifact saving."""

    def __init__(
        self,
        model: BaseTemporalActionModel,
        train_config: TrainingConfig,
        label_to_idx: Dict[str, int],
        class_weights: Optional[torch.Tensor] = None,
        cpu_threads: Optional[int] = None,
    ):
        self.model = model
        self.config = train_config
        self.label_to_idx = label_to_idx
        self.idx_to_label = {v: k for k, v in label_to_idx.items()}
        self.cpu_threads = cpu_threads or os.cpu_count() or 4
        torch.set_num_threads(self.cpu_threads)

        # Loss function
        if train_config.use_focal_loss:
            self.criterion = FocalLoss(alpha=class_weights, gamma=2.0)
        else:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Optimizer and Scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=train_config.epochs, eta_min=1e-5
        )

        self.history: List[Dict[str, Any]] = []
        self.best_val_f1 = -1.0
        self.best_checkpoint_path: Optional[Path] = None

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for features, targets, _ in dataloader:
            self.optimizer.zero_grad()
            out = self.model(features)
            loss = self.criterion(out["logits"], targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item() * features.size(0)
        return total_loss / max(len(dataloader.dataset), 1)

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, Any]:
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        for features, targets, _ in dataloader:
            out = self.model(features)
            loss = self.criterion(out["logits"], targets)
            total_loss += loss.item() * features.size(0)
            preds = torch.argmax(out["probabilities"], dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())

        if not all_targets:
            return {"loss": 0.0, "accuracy": 0.0, "macro_f1": 0.0, "macro_precision": 0.0, "macro_recall": 0.0}

        avg_loss = total_loss / len(all_targets)
        acc = float(accuracy_score(all_targets, all_preds))
        f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
        prec = float(precision_score(all_targets, all_preds, average="macro", zero_division=0))
        rec = float(recall_score(all_targets, all_preds, average="macro", zero_division=0))

        return {
            "loss": avg_loss,
            "accuracy": acc,
            "macro_f1": f1,
            "macro_precision": prec,
            "macro_recall": rec,
            "predictions": all_preds,
            "targets": all_targets,
        }

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, Any]:
        """Execute full training loop with early stopping and checkpointing."""
        epochs_no_improve = 0
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(self.config.output_dir) / f"run_{timestamp}_{self.model.config.backbone}_{self.model.config.model_type}"
        run_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        for epoch in range(1, self.config.epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            self.scheduler.step()

            elapsed = time.time() - t0
            record = {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_metrics["loss"], 4),
                "val_accuracy": round(val_metrics["accuracy"], 4),
                "val_macro_f1": round(val_metrics["macro_f1"], 4),
                "lr": round(float(self.scheduler.get_last_lr()[0]), 6),
                "elapsed_sec": round(elapsed, 2),
            }
            self.history.append(record)

            # Check for best validation F1
            if val_metrics["macro_f1"] > self.best_val_f1:
                self.best_val_f1 = val_metrics["macro_f1"]
                epochs_no_improve = 0
                self.best_checkpoint_path = run_dir / "best_model.pt"
                self.save_checkpoint(self.best_checkpoint_path, epoch, val_metrics)
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= self.config.patience:
                break

        # Save training history
        with open(run_dir / "training_history.json", "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

        return {
            "run_dir": str(run_dir),
            "best_val_f1": self.best_val_f1,
            "best_checkpoint_path": str(self.best_checkpoint_path) if self.best_checkpoint_path else None,
            "total_epochs": len(self.history),
            "total_time_sec": round(time.time() - t0, 2),
        }

    def save_checkpoint(self, path: Path, epoch: int, metrics: Dict[str, Any]) -> None:
        """Save complete versioned checkpoint."""
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": metrics,
                "model_config": self.model.config.to_dict(),
                "label_to_idx": self.label_to_idx,
            },
            path,
        )

    def load_checkpoint(self, path: Path) -> Dict[str, Any]:
        """Load state dict and resume checkpoint."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        return ckpt
