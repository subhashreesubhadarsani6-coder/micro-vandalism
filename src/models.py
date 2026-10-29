"""
Egocentric Temporal Action Recognition Models for Heritage Protection AI.
Includes lightweight CPU-friendly architectures:
- TemporalConv1DNet (Main lightweight 1D temporal convolutional network)
- TemporalPoolingBaseline (Temporal mean & max MLP baseline)
- TemporalGRUNet (Recurrent sequence comparison baseline)
- SklearnBaselineWrapper (Legacy Logistic Regression baseline)
"""
from dataclasses import dataclass, field, asdict
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import CANDIDATE_ACTION_CLASSES
from src.feature_extractor import BACKBONE_FEATURE_DIMS


# Default heritage wall risk severity mapping for the 11 candidate classes.
# High-severity physical alteration actions map near 1.0; benign tourist actions map to 0.0.
# NOTE: This defines ACTION RISK PROBABILITY, not physical damage measurement.
DEFAULT_ACTION_RISK_WEIGHTS: Dict[str, float] = {
    "potential_chipping": 1.00,
    "potential_scratching": 0.95,
    "potential_scraping": 0.90,
    "potential_marking": 0.85,
    "ambiguous_interaction": 0.50,
    "inspection": 0.15,
    "permitted_touching": 0.10,
    "pointing": 0.00,
    "photographing": 0.00,
    "walking": 0.00,
    "normal_looking": 0.00,
    "unverified": 0.20,
}


@dataclass
class ModelConfig:
    """Serializable, reproducible configuration for temporal action recognition models."""
    model_type: str = "conv1d"                      # 'conv1d', 'pooling_mlp', 'gru'
    backbone: str = "mobilenet_v3_small"            # 'mobilenet_v3_small', 'resnet18', 'mobilenet_v3_large', 'custom'
    input_dim: Optional[int] = None                 # Defaults to backbone dimension if None
    num_frames: int = 8                             # Number of temporal frame samples
    num_classes: int = len(CANDIDATE_ACTION_CLASSES)# Class output dimension
    hidden_dim: int = 128                           # Hidden channel dimension
    dropout: float = 0.3                            # Regularization dropout probability
    class_names: List[str] = field(default_factory=lambda: sorted(list(CANDIDATE_ACTION_CLASSES)))
    risk_weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_ACTION_RISK_WEIGHTS))

    def __post_init__(self):
        # Auto-align input_dim to backbone if input_dim is not explicitly provided
        if self.input_dim is None:
            self.input_dim = BACKBONE_FEATURE_DIMS.get(self.backbone, 576)
        if self.num_classes != len(self.class_names):
            self.num_classes = len(self.class_names)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        return cls(**data)

    def save_json(self, path: Path) -> None:
        p = Path(path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: Path) -> "ModelConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


class BaseTemporalActionModel(nn.Module):
    """Abstract base class for video temporal action models with risk mapping and tensor validation."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.num_classes = config.num_classes
        self.num_frames = config.num_frames

        # Construct fixed risk weight vector matching class_names order
        risk_vector = [config.risk_weights.get(c, 0.0) for c in config.class_names]
        self.register_buffer("risk_vector", torch.tensor(risk_vector, dtype=torch.float32))

    def check_input_tensor(self, x: torch.Tensor) -> None:
        """Validate input tensor shape (B, T, D)."""
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor input, got {type(x)}")
        if x.ndim != 3:
            raise ValueError(
                f"Expected 3D input tensor of shape (B, T, D), got {x.ndim}D tensor with shape {list(x.shape)}"
            )
        b, t, d = x.shape
        if b <= 0:
            raise ValueError(f"Batch size must be positive, got {b}")
        if t < 1:
            raise ValueError(f"Frame count T must be at least 1, got {t}")
        if d != self.input_dim:
            raise ValueError(
                f"Embedding dimension mismatch: model expects D={self.input_dim} (for {self.config.backbone}), "
                f"but received input with D={d}."
            )

    def compute_risk_score(self, probabilities: torch.Tensor) -> torch.Tensor:
        """
        Compute continuous action risk score in [0.0, 1.0] by taking the dot product
        of predicted action probabilities and domain risk severity weights.
        """
        return torch.matmul(probabilities, self.risk_vector)


class TemporalPoolingBaseline(BaseTemporalActionModel):
    """
    Temporal baseline model: computes mean and max pooling across ordered frame embeddings,
    concatenates into a 2D-dim vector, and passes through a regularized MLP classifier.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        pooled_in_dim = config.input_dim * 2  # Mean + Max concatenation

        self.classifier = nn.Sequential(
            nn.Linear(pooled_in_dim, config.hidden_dim),
            nn.BatchNorm1d(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(p=config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(p=config.dropout / 2.0),
            nn.Linear(config.hidden_dim // 2, config.num_classes),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Tensor of shape (B, T, D)
        Returns:
            Dictionary containing 'logits', 'probabilities', and 'risk_score'
        """
        self.check_input_tensor(x)
        # x is (B, T, D)
        x_mean = x.mean(dim=1)  # (B, D)
        x_max = x.max(dim=1)[0] # (B, D)
        pooled = torch.cat([x_mean, x_max], dim=1)  # (B, 2D)

        logits = self.classifier(pooled)
        probabilities = F.softmax(logits, dim=-1)
        risk_score = self.compute_risk_score(probabilities)

        return {
            "logits": logits,
            "probabilities": probabilities,
            "risk_score": risk_score,
            "features": pooled,
        }


class TemporalConv1DNet(BaseTemporalActionModel):
    """
    Main Lightweight Temporal Model: 1D Temporal Convolutional Network.
    Preserves and exploits temporal frame order via temporal convolutions, residual skips,
    and temporal attentive pooling to detect dynamic tool/hand micro-motions.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        h = config.hidden_dim

        # Input projection and first temporal convolution
        self.in_proj = nn.Conv1d(config.input_dim, h, kernel_size=1)
        self.conv1 = nn.Conv1d(h, h, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(h)

        # Second dilated temporal convolution for broader temporal receptive field
        self.conv2 = nn.Conv1d(h, h, kernel_size=3, padding=2, dilation=2, bias=False)
        self.bn2 = nn.BatchNorm1d(h)

        self.act = nn.GELU()
        self.dropout = nn.Dropout(p=config.dropout)

        # Attentive temporal pooling over the time dimension T
        self.temporal_attn = nn.Sequential(
            nn.Linear(h, h // 2),
            nn.Tanh(),
            nn.Linear(h // 2, 1),
        )

        # Final classification head: combines attentive pool (h) + temporal max pool (h) -> 2h
        self.head = nn.Sequential(
            nn.Linear(h * 2, h),
            nn.GELU(),
            nn.Dropout(p=config.dropout),
            nn.Linear(h, config.num_classes),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Tensor of shape (B, T, D)
        Returns:
            Dictionary containing 'logits', 'probabilities', and 'risk_score'
        """
        self.check_input_tensor(x)
        # Transpose for Conv1d: (B, T, D) -> (B, D, T)
        x_t = x.transpose(1, 2)
        res = self.in_proj(x_t)  # (B, H, T)

        h = self.conv1(res)
        h = self.bn1(h)
        h = self.act(h)
        h = self.dropout(h)

        h = self.conv2(h)
        h = self.bn2(h)
        h = self.act(h + res)  # Residual connection
        h = self.dropout(h)    # (B, H, T)

        # Temporal attention over time: (B, T, H)
        h_seq = h.transpose(1, 2)
        attn_scores = self.temporal_attn(h_seq)       # (B, T, 1)
        attn_weights = F.softmax(attn_scores, dim=1)  # (B, T, 1)
        pooled_attn = (h_seq * attn_weights).sum(dim=1) # (B, H)

        # Temporal max pooling
        pooled_max = h.max(dim=2)[0]                  # (B, H)

        representation = torch.cat([pooled_attn, pooled_max], dim=1)  # (B, 2H)
        logits = self.head(representation)
        probabilities = F.softmax(logits, dim=-1)
        risk_score = self.compute_risk_score(probabilities)

        return {
            "logits": logits,
            "probabilities": probabilities,
            "risk_score": risk_score,
            "temporal_features": representation,
            "attention_weights": attn_weights.squeeze(-1),
        }


class TemporalGRUNet(BaseTemporalActionModel):
    """
    Comparison Recurrent Model: Lightweight Bi-directional GRU over ordered frame embeddings.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        h = config.hidden_dim // 2  # Bidirectional doubles hidden state

        self.gru = nn.GRU(
            input_size=config.input_dim,
            hidden_size=h,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(p=config.dropout)
        # Bidirectional GRU produces 2 * h = hidden_dim features
        self.classifier = nn.Sequential(
            nn.Linear(h * 2, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(p=config.dropout),
            nn.Linear(config.hidden_dim, config.num_classes),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.check_input_tensor(x)
        out, _ = self.gru(x)  # (B, T, 2h)
        # Combine temporal mean and temporal last hidden state
        gru_mean = out.mean(dim=1)
        gru_last = out[:, -1, :]
        rep = self.dropout(gru_mean + gru_last)

        logits = self.classifier(rep)
        probabilities = F.softmax(logits, dim=-1)
        risk_score = self.compute_risk_score(probabilities)

        return {
            "logits": logits,
            "probabilities": probabilities,
            "risk_score": risk_score,
            "features": rep,
        }


def build_model(config: ModelConfig) -> BaseTemporalActionModel:
    """Factory function to build the requested temporal model from config."""
    m_type = config.model_type.lower()
    if m_type in ("conv1d", "temporal_conv", "tcn"):
        return TemporalConv1DNet(config)
    elif m_type in ("pooling_mlp", "baseline", "pooling"):
        return TemporalPoolingBaseline(config)
    elif m_type in ("gru", "recurrent"):
        return TemporalGRUNet(config)
    else:
        raise ValueError(
            f"Unknown model_type: '{config.model_type}'. Must be one of ['conv1d', 'pooling_mlp', 'gru']"
        )


# =====================================================================
# Loss Functions for Class Imbalance
# =====================================================================

def compute_class_weights(
    labels: np.ndarray,
    num_classes: int,
    method: str = "inverse_frequency",
    beta: float = 0.999,
) -> torch.Tensor:
    """
    Compute class weighting vector to counter severe dataset imbalance.
    Supports:
        - 'inverse_frequency': w_c = N / (C * count_c)
        - 'effective_samples': w_c = (1 - beta) / (1 - beta^{count_c}) (Cui et al., CVPR 2019)
    """
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)  # Avoid division by zero

    if method == "inverse_frequency":
        total = np.sum(counts)
        weights = total / (num_classes * counts)
    elif method == "effective_samples":
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / np.maximum(effective_num, 1e-6)
    else:
        weights = np.ones(num_classes, dtype=np.float32)

    # Normalize weights so mean is 1.0
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance:
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


# =====================================================================
# Legacy Comparison Baseline (Scikit-Learn Logistic Regression)
# =====================================================================

class SklearnBaselineWrapper:
    """
    Encapsulates the legacy Logistic Regression pipeline from vandalism_cpu.py.
    Operates on flattened/pooled features (mean+max pooling) with StandardScaler.
    """

    def __init__(self, c_param: float = 1.0, class_weight: str = "balanced", seed: int = 42):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        self.scaler = StandardScaler()
        self.clf = LogisticRegression(
            C=c_param,
            class_weight=class_weight,
            max_iter=5000,
            random_state=seed,
        )
        self.is_fitted = False
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SklearnBaselineWrapper":
        # X: (N, 2D) pooled features
        X_scaled = self.scaler.fit_transform(X)
        self.clf.fit(X_scaled, y)
        self.is_fitted = True
        self.classes_ = self.clf.classes_
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        X_scaled = self.scaler.transform(X)
        return self.clf.predict_proba(X_scaled)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        X_scaled = self.scaler.transform(X)
        return self.clf.predict(X_scaled)
