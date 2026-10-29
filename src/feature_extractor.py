"""
Pretrained feature extraction module supporting MobileNetV3-Small, ResNet18, and MobileNetV3-Large.
Handles parameter freezing, CPU thread allocation, batched inference, and weight availability.
"""
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import numpy as np
import torch
import torchvision


SUPPORTED_BACKBONES = ("mobilenet_v3_small", "resnet18", "mobilenet_v3_large")

# Expected per-frame raw feature embedding dimensions (before temporal pooling)
BACKBONE_FEATURE_DIMS = {
    "mobilenet_v3_small": 576,
    "resnet18": 512,
    "mobilenet_v3_large": 960,
}

# Pretrained weight URLs and sizes for transparency
WEIGHTS_METADATA = {
    "mobilenet_v3_small": {
        "url": "https://download.pytorch.org/models/mobilenet_v3_small-047dc642.pth",
        "file_name": "mobilenet_v3_small-047dc642.pth",
        "size_mb": 9.8,
    },
    "resnet18": {
        "url": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
        "file_name": "resnet18-f37072fd.pth",
        "size_mb": 44.7,
    },
    "mobilenet_v3_large": {
        "url": "https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth",
        "file_name": "mobilenet_v3_large-8738ca79.pth",
        "size_mb": 21.1,
    },
}


class PretrainedWeightsUnavailableError(Exception):
    """Raised when pretrained weights are not cached locally and automatic downloads are disabled."""
    pass


class FeatureExtractor:
    """
    Modular frozen feature extractor for video action recognition.
    Extracts per-frame feature vectors and computes temporal pooled representations.
    """

    def __init__(
        self,
        backbone_name: str = "resnet18",
        weights: Optional[str] = "none",
        allow_download: bool = False,
        custom_weights_path: Optional[Path] = None,
        cpu_threads: Optional[int] = None,
    ):
        backbone_name = backbone_name.lower()
        if backbone_name not in SUPPORTED_BACKBONES:
            raise ValueError(
                f"Unsupported backbone: '{backbone_name}'. Must be one of {SUPPORTED_BACKBONES}"
            )

        self.backbone_name = backbone_name
        self.weights_mode = str(weights).lower() if weights else "none"
        self.allow_download = allow_download
        self.custom_weights_path = Path(custom_weights_path) if custom_weights_path else None
        self.per_frame_dim = BACKBONE_FEATURE_DIMS[backbone_name]
        self.pooled_dim = self.per_frame_dim * 2  # Concat of temporal mean and max pooling

        # Configure CPU threading
        threads = cpu_threads or os.cpu_count() or 4
        torch.set_num_threads(threads)

        self.model = self._build_model()
        self.model.eval()
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def _check_local_weights(self) -> Optional[Path]:
        """Check if pretrained weights are available in local torch cache or custom path."""
        if self.custom_weights_path and self.custom_weights_path.exists():
            return self.custom_weights_path

        meta = WEIGHTS_METADATA.get(self.backbone_name)
        if not meta:
            return None

        hub_checkpoints = Path(torch.hub.get_dir()) / "checkpoints" / meta["file_name"]
        if hub_checkpoints.exists():
            return hub_checkpoints

        return None

    def _build_model(self) -> torch.nn.Module:
        """Construct the backbone architecture and apply weight initialization."""
        use_pretrained = self.weights_mode in ("pretrained", "default", "imagenet")

        if use_pretrained:
            local_weights = self._check_local_weights()
            if local_weights is None and not self.allow_download:
                meta = WEIGHTS_METADATA[self.backbone_name]
                raise PretrainedWeightsUnavailableError(
                    f"Pretrained weights for '{self.backbone_name}' are not found locally.\n"
                    f"File: {meta['file_name']} (~{meta['size_mb']} MB)\n"
                    f"Download URL: {meta['url']}\n"
                    f"To prevent unauthorized data transfer, automatic downloads are disabled by default.\n"
                    f"Solutions:\n"
                    f"  1. Pass allow_download=True to permit torchvision to download this weight file.\n"
                    f"  2. Provide a local checkpoint via custom_weights_path.\n"
                    f"  3. Use weights='none' for unweighted architecture initialization (tests/benchmarks)."
                )

        # 1. ResNet18
        if self.backbone_name == "resnet18":
            if use_pretrained and self.allow_download:
                model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
            elif use_pretrained and local_weights:
                model = torchvision.models.resnet18(weights=None)
                state = torch.load(local_weights, map_location="cpu", weights_only=True)
                model.load_state_dict(state)
            else:
                model = torchvision.models.resnet18(weights=None)

            # Truncate fc layer to output raw 512-dim embedding
            model.fc = torch.nn.Identity()
            return model

        # 2. MobileNetV3-Small
        elif self.backbone_name == "mobilenet_v3_small":
            if use_pretrained and self.allow_download:
                model = torchvision.models.mobilenet_v3_small(
                    weights=torchvision.models.MobileNet_V3_Small_Weights.DEFAULT
                )
            elif use_pretrained and local_weights:
                model = torchvision.models.mobilenet_v3_small(weights=None)
                state = torch.load(local_weights, map_location="cpu", weights_only=True)
                model.load_state_dict(state)
            else:
                model = torchvision.models.mobilenet_v3_small(weights=None)

            # MobileNetV3 structure: features -> avgpool -> classifier
            # Replace classifier with Identity to output 576-dim embedding
            model.classifier = torch.nn.Identity()
            return model

        # 3. MobileNetV3-Large
        elif self.backbone_name == "mobilenet_v3_large":
            if use_pretrained and self.allow_download:
                model = torchvision.models.mobilenet_v3_large(
                    weights=torchvision.models.MobileNet_V3_Large_Weights.DEFAULT
                )
            elif use_pretrained and local_weights:
                model = torchvision.models.mobilenet_v3_large(weights=None)
                state = torch.load(local_weights, map_location="cpu", weights_only=True)
                model.load_state_dict(state)
            else:
                model = torchvision.models.mobilenet_v3_large(weights=None)

            model.classifier = torch.nn.Identity()
            return model

        raise ValueError(f"Unknown backbone: {self.backbone_name}")

    def extract(self, frames_tensor: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features from a batch of frames (N, 3, H, W).
        Returns:
            per_frame_feats: np.ndarray of shape (N, per_frame_dim)
            pooled_feat: np.ndarray of shape (pooled_dim,) [mean + max pooling]
        """
        if frames_tensor.ndim != 4 or frames_tensor.shape[1] != 3:
            raise ValueError(
                f"Expected frames_tensor of shape (N, 3, H, W), got {frames_tensor.shape}"
            )

        with torch.no_grad():
            out = self.model(frames_tensor)
            # Flatten if needed (e.g. Identity on (N, D, 1, 1))
            if out.ndim > 2:
                out = torch.flatten(out, 1)
            per_frame_feats = out.cpu().numpy().astype(np.float32)

        # Temporal pooling across the N frames: mean + max pooling
        mean_pool = per_frame_feats.mean(axis=0)
        max_pool = per_frame_feats.max(axis=0)
        pooled_feat = np.concatenate([mean_pool, max_pool]).astype(np.float32)

        return per_frame_feats, pooled_feat

    def get_config_dict(self) -> Dict[str, Any]:
        return {
            "backbone": self.backbone_name,
            "weights_mode": self.weights_mode,
            "per_frame_dim": self.per_frame_dim,
            "pooled_dim": self.pooled_dim,
            "pooling_method": "mean_max_concat",
        }
