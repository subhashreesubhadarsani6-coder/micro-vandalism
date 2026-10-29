"""
Dataset Management Module for Guarding the Stones.
Handles safe archive extraction, file scanning, manifest generation (JSONL & CSV),
duplicate detection, validation audits, and statistical summaries.
"""
import csv
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
import zipfile

from src.config import (
    DatasetConfig,
    SUPPORTED_VIDEO_EXTENSIONS,
    CANDIDATE_ACTION_CLASSES,
    CAMERA_VIEWPOINTS,
)
from src.video_parser import extract_video_metadata, VideoMetadata
from src.annotation_schema import VideoAnnotation


@dataclass
class DatasetRecord:
    """Consolidated manifest record joining video technical metadata and annotation."""
    video_id: str
    file_path: str
    file_name: str
    format: str
    file_size_bytes: int
    sha256: str
    duration_sec: Optional[float]
    frame_count: Optional[int]
    fps: Optional[float]
    width: Optional[int]
    height: Optional[int]
    action_label: str
    camera_viewpoint: str
    is_egocentric: bool
    watermark_detected: bool
    validation_status: str
    source_archive: Optional[str] = None
    participant_id: Optional[str] = None
    session_id: Optional[str] = None
    annotator_id: Optional[str] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_id": self.video_id,
            "file_path": self.file_path,
            "file_name": self.file_name,
            "format": self.format,
            "file_size_bytes": self.file_size_bytes,
            "sha256": self.sha256,
            "duration_sec": self.duration_sec,
            "frame_count": self.frame_count,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "action_label": self.action_label,
            "camera_viewpoint": self.camera_viewpoint,
            "is_egocentric": self.is_egocentric,
            "watermark_detected": self.watermark_detected,
            "validation_status": self.validation_status,
            "source_archive": self.source_archive,
            "participant_id": self.participant_id,
            "session_id": self.session_id,
            "annotator_id": self.annotator_id,
            "notes": self.notes,
        }


class DatasetManager:
    """Manages archive extraction, video auditing, manifest generation, and validation."""

    def __init__(self, config: Optional[DatasetConfig] = None):
        self.config = config or DatasetConfig()
        self.config.ensure_directories()

    def extract_archive(
        self,
        archive_path: Optional[Path] = None,
        target_dir: Optional[Path] = None,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Extract video files from a ZIP archive without overwriting any existing files.
        """
        src_zip = Path(archive_path or self.config.archive_path).resolve()
        dest_dir = Path(target_dir or self.config.data_dir).resolve()

        if not src_zip.exists():
            return {
                "success": False,
                "error": f"Archive does not exist: {src_zip}",
                "extracted_count": 0,
                "skipped_count": 0,
            }

        dest_dir.mkdir(parents=True, exist_ok=True)
        extracted = 0
        skipped = 0
        extracted_files: List[str] = []

        with zipfile.ZipFile(src_zip, "r") as zf:
            for member in zf.infolist():
                if member.is_dir():
                    continue

                member_name = member.filename
                # Security: prevent directory traversal
                clean_name = os.path.normpath(member_name).lstrip("/\\")
                if ".." in clean_name.split(os.path.sep):
                    continue

                target_file = dest_dir / clean_name
                if target_file.exists():
                    skipped += 1
                    continue

                if not dry_run:
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(member, "r") as src, open(target_file, "wb") as dst:
                        # Stream copy in 64 KB chunks
                        while chunk := src.read(65536):
                            dst.write(chunk)

                extracted += 1
                extracted_files.append(str(target_file))

        return {
            "success": True,
            "archive_path": str(src_zip),
            "target_dir": str(dest_dir),
            "extracted_count": extracted,
            "skipped_count": skipped,
            "dry_run": dry_run,
            "extracted_files": extracted_files[:10],
        }

    def scan_videos(self, target_dir: Optional[Path] = None) -> List[Path]:
        """Scan directory recursively for all supported video extensions."""
        scan_dir = Path(target_dir or self.config.data_dir).resolve()
        if not scan_dir.exists():
            return []

        videos: List[Path] = []
        for root, _, files in os.walk(scan_dir):
            for file in sorted(files):
                ext = Path(file).suffix.lower()
                if ext in self.config.supported_extensions:
                    videos.append(Path(root) / file)
        return videos

    def build_manifest(
        self,
        target_dir: Optional[Path] = None,
        annotations_file: Optional[Path] = None,
        source_archive_name: Optional[str] = None,
    ) -> List[DatasetRecord]:
        """
        Inspect all videos, cross-reference annotations, check duplicates, and construct records.
        """
        video_paths = self.scan_videos(target_dir)
        records: List[DatasetRecord] = []
        sha_registry: Dict[str, str] = {}  # sha256 -> first video_id

        # Load existing annotations if available
        existing_annots: Dict[str, VideoAnnotation] = {}
        if annotations_file and Path(annotations_file).exists():
            with open(annotations_file, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
                for item in raw_data:
                    ann = VideoAnnotation.from_dict(item)
                    existing_annots[ann.video_id] = ann

        for idx, vpath in enumerate(video_paths):
            meta = extract_video_metadata(vpath)
            vid = f"VID_{meta.sha256[:12]}" if meta.sha256 else f"VID_ERR_{idx:04d}"

            # Check duplicate by hash
            status = meta.validation_status
            if meta.sha256:
                if meta.sha256 in sha_registry:
                    status = "duplicate"
                else:
                    sha_registry[meta.sha256] = vid

            # Default values if no human annotation is present
            action_label = "unverified"
            camera_viewpoint = "unknown"
            is_egocentric = False
            watermark_detected = False
            participant_id = None
            session_id = None
            annotator_id = None
            notes = meta.error_message or ""

            # Check for known stock footage provenance in legacy set
            path_str = str(vpath).lower()
            if "carving" in path_str or "normal" in path_str:
                camera_viewpoint = "stock_footage"
                watermark_detected = True
                notes += "Legacy stock clip; observed iStock watermark; third-person perspective."

            # Merge with verified human annotation if present
            if vid in existing_annots:
                ann = existing_annots[vid]
                action_label = ann.action_label
                camera_viewpoint = ann.camera_viewpoint
                is_egocentric = ann.is_egocentric
                watermark_detected = ann.watermark_detected
                annotator_id = ann.annotator_id
                if ann.notes:
                    notes = f"{notes} | {ann.notes}".strip(" |")

            # Store relative path where possible for portability
            try:
                rel_path = str(vpath.relative_to(self.config.data_dir.parent))
            except ValueError:
                rel_path = str(vpath)

            record = DatasetRecord(
                video_id=vid,
                file_path=rel_path,
                file_name=vpath.name,
                format=meta.format,
                file_size_bytes=meta.file_size_bytes,
                sha256=meta.sha256,
                duration_sec=meta.duration_sec,
                frame_count=meta.frame_count,
                fps=meta.fps,
                width=meta.width,
                height=meta.height,
                action_label=action_label,
                camera_viewpoint=camera_viewpoint,
                is_egocentric=is_egocentric,
                watermark_detected=watermark_detected,
                validation_status=status,
                source_archive=source_archive_name or self.config.archive_path.name,
                participant_id=participant_id,
                session_id=session_id,
                annotator_id=annotator_id,
                notes=notes,
            )
            records.append(record)

        return records

    def export_manifest(
        self,
        records: List[DatasetRecord],
        out_base_path: Optional[Path] = None
    ) -> Dict[str, str]:
        """Export records to both JSONL and CSV formats."""
        base = Path(out_base_path or (self.config.annotations_dir / "manifest")).resolve()
        base.parent.mkdir(parents=True, exist_ok=True)
        jsonl_path = base.with_suffix(".jsonl")
        csv_path = base.with_suffix(".csv")

        # Export JSONL
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r.to_dict()) + "\n")

        # Export CSV
        if records:
            fieldnames = list(records[0].to_dict().keys())
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in records:
                    writer.writerow(r.to_dict())
        else:
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                pass

        return {"jsonl": str(jsonl_path), "csv": str(csv_path)}

    def generate_summary(self, records: List[DatasetRecord]) -> Dict[str, Any]:
        """Generate comprehensive statistics, class counts, and detected problems."""
        total = len(records)
        statuses: Dict[str, int] = {}
        viewpoints: Dict[str, int] = {}
        action_labels: Dict[str, int] = {}
        formats: Dict[str, int] = {}
        watermarks = 0
        egocentric_count = 0
        total_duration = 0.0
        total_frames = 0
        problems: List[str] = []

        for r in records:
            statuses[r.validation_status] = statuses.get(r.validation_status, 0) + 1
            viewpoints[r.camera_viewpoint] = viewpoints.get(r.camera_viewpoint, 0) + 1
            action_labels[r.action_label] = action_labels.get(r.action_label, 0) + 1
            formats[r.format] = formats.get(r.format, 0) + 1
            if r.watermark_detected:
                watermarks += 1
            if r.is_egocentric:
                egocentric_count += 1
            if r.duration_sec:
                total_duration += r.duration_sec
            if r.frame_count:
                total_frames += r.frame_count

        # Detect high-level data problems
        if total == 0:
            problems.append("Dataset is empty. No video files found.")
        if statuses.get("empty", 0) > 0:
            problems.append(f"Found {statuses['empty']} zero-byte (empty) video files.")
        if statuses.get("corrupt", 0) > 0 or statuses.get("unreadable", 0) > 0:
            problems.append(f"Found {statuses.get('corrupt', 0) + statuses.get('unreadable', 0)} unreadable/corrupt files.")
        if statuses.get("duplicate", 0) > 0:
            problems.append(f"Found {statuses['duplicate']} duplicate video clips (identical SHA-256).")
        if watermarks > 0:
            problems.append(f"Found {watermarks} clips with commercial watermarks (Clever Hans shortcut risk).")
        if egocentric_count == 0 and total > 0:
            problems.append("CRITICAL: Zero clips possess a validated first-person (egocentric) perspective.")
        if action_labels.get("unverified", 0) == total and total > 0:
            problems.append("All action labels are 'unverified'. Manual human annotation is required.")
        if "normal_looking" not in action_labels and "walking" not in action_labels:
            problems.append("Negative/normal baseline classes are missing from dataset.")

        return {
            "total_videos": total,
            "total_duration_sec": round(total_duration, 2),
            "total_duration_hours": round(total_duration / 3600.0, 3),
            "total_frames": total_frames,
            "validation_status_breakdown": statuses,
            "camera_viewpoint_breakdown": viewpoints,
            "action_label_breakdown": action_labels,
            "format_breakdown": formats,
            "watermark_count": watermarks,
            "egocentric_count": egocentric_count,
            "detected_problems": problems,
        }
