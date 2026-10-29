"""
Configuration module for Guarding the Stones dataset preparation pipeline.
Provides configurable paths via CLI arguments and environment variables.
"""
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import FrozenSet, Set

# Candidate action classes as specified in requirements
CANDIDATE_ACTION_CLASSES: FrozenSet[str] = frozenset([
    "normal_looking",
    "walking",
    "pointing",
    "photographing",
    "permitted_touching",
    "inspection",
    "potential_scratching",
    "potential_scraping",
    "potential_chipping",
    "potential_marking",
    "ambiguous_interaction",
])

# Supported video file extensions
SUPPORTED_VIDEO_EXTENSIONS: FrozenSet[str] = frozenset([
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
])

# Recognized camera viewpoints
CAMERA_VIEWPOINTS: FrozenSet[str] = frozenset([
    "egocentric",       # First-person (head, chest, smart-glasses mount)
    "third_person",     # Handheld / bystander camera
    "surveillance_cctv",# Fixed elevated surveillance angle
    "overhead",         # Directly looking down from ceiling/crane
    "drone",            # Aerial moving platform
    "stock_footage",    # Commercial stock video with watermarks/staged
    "unknown",          # Unverified viewpoint
])

# Validation status codes
VALIDATION_STATUSES: FrozenSet[str] = frozenset([
    "valid",
    "unreadable",
    "empty",
    "corrupt",
    "duplicate",
    "viewpoint_mismatch",
    "missing_label",
    "unverified",
])

# Default paths
DEFAULT_WORKSPACE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = DEFAULT_WORKSPACE_DIR / "data"
DEFAULT_RAW_DIR = DEFAULT_DATA_DIR / "raw"
DEFAULT_PROCESSED_DIR = DEFAULT_DATA_DIR / "processed"
DEFAULT_ANNOTATIONS_DIR = DEFAULT_DATA_DIR / "annotations"
DEFAULT_ARCHIVE_PATH = DEFAULT_WORKSPACE_DIR / "carving.zip"


def get_dataset_dir(cli_path: str = None) -> Path:
    """
    Resolve dataset directory with precedence:
    1. CLI argument (if provided)
    2. Environment variable GUARDING_DATASET_DIR or VANDALISM_DATASET_DIR
    3. Default workspace data/raw directory
    """
    if cli_path:
        return Path(cli_path).resolve()
    
    env_dir = os.environ.get("GUARDING_DATASET_DIR") or os.environ.get("VANDALISM_DATASET_DIR")
    if env_dir:
        return Path(env_dir).resolve()
        
    return DEFAULT_RAW_DIR


def get_archive_path(cli_path: str = None) -> Path:
    """
    Resolve archive path with precedence:
    1. CLI argument (if provided)
    2. Environment variable GUARDING_ARCHIVE_PATH
    3. Default workspace carving.zip or Downloads fallback
    """
    if cli_path:
        return Path(cli_path).resolve()
        
    env_archive = os.environ.get("GUARDING_ARCHIVE_PATH")
    if env_archive:
        return Path(env_archive).resolve()
        
    if DEFAULT_ARCHIVE_PATH.exists():
        return DEFAULT_ARCHIVE_PATH
        
    # User Downloads fallback
    downloads_archive = Path.home() / "Downloads" / "carving.zip"
    if downloads_archive.exists():
        return downloads_archive
        
    return DEFAULT_ARCHIVE_PATH


@dataclass
class DatasetConfig:
    """Runtime dataset pipeline configuration."""
    data_dir: Path = field(default_factory=get_dataset_dir)
    archive_path: Path = field(default_factory=get_archive_path)
    annotations_dir: Path = field(default_factory=lambda: DEFAULT_ANNOTATIONS_DIR)
    supported_extensions: Set[str] = field(default_factory=lambda: set(SUPPORTED_VIDEO_EXTENSIONS))
    candidate_classes: Set[str] = field(default_factory=lambda: set(CANDIDATE_ACTION_CLASSES))
    
    def ensure_directories(self) -> None:
        """Create necessary project data directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.annotations_dir.mkdir(parents=True, exist_ok=True)
