"""
Annotation schema and validation engine for first-person heritage protection action recognition.
Defines candidate classes, timestamp bounds, camera viewpoint flags, and validation logic.
"""
from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from src.config import CANDIDATE_ACTION_CLASSES, CAMERA_VIEWPOINTS


# Comprehensive JSON Schema definition for external tools (CVAT, Label Studio, custom UI)
ANNOTATION_JSON_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "EgocentricHeritageVandalismAnnotation",
    "description": "Annotation record for video action recognition in heritage wall protection",
    "type": "object",
    "required": [
        "video_id",
        "file_name",
        "action_label",
        "camera_viewpoint",
        "is_egocentric",
        "verified_by_human"
    ],
    "properties": {
        "video_id": {
            "type": "string",
            "description": "Unique identifier for the video clip (e.g. SHA-256 or clip ID)"
        },
        "file_name": {
            "type": "string",
            "description": "Base filename of the video clip"
        },
        "action_label": {
            "type": "string",
            "enum": list(sorted(CANDIDATE_ACTION_CLASSES)) + ["unverified", "ambiguous_interaction"],
            "description": "Action class according to the heritage protection taxonomy"
        },
        "start_sec": {
            "type": ["number", "null"],
            "minimum": 0.0,
            "description": "Optional start timestamp in seconds for the action event"
        },
        "end_sec": {
            "type": ["number", "null"],
            "minimum": 0.0,
            "description": "Optional end timestamp in seconds for the action event"
        },
        "is_egocentric": {
            "type": "boolean",
            "description": "True if filmed from a first-person / body-worn camera perspective"
        },
        "camera_viewpoint": {
            "type": "string",
            "enum": list(sorted(CAMERA_VIEWPOINTS)),
            "description": "Observed viewpoint angle"
        },
        "watermark_detected": {
            "type": "boolean",
            "description": "Whether a stock or camera watermark is present in the frame"
        },
        "tool_used": {
            "type": ["string", "null"],
            "enum": [
                "none",
                "key",
                "coin",
                "pocket_knife",
                "screwdriver",
                "metal_stylus",
                "marker_pen",
                "spray_can",
                "chalk",
                "hand_finger",
                "camera_phone",
                "other"
            ],
            "description": "Object or tool involved in the physical interaction"
        },
        "target_surface": {
            "type": ["string", "null"],
            "enum": [
                "sandstone",
                "granite",
                "limestone",
                "marble",
                "brick",
                "plaster",
                "painted_mural",
                "unknown",
                "not_applicable"
            ],
            "description": "Type of architectural or monument substrate"
        },
        "verified_by_human": {
            "type": "boolean",
            "description": "Must be true; labels must never be assigned based on filename alone"
        },
        "annotator_id": {
            "type": ["string", "null"],
            "description": "Identifier of the human annotator or domain reviewer"
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Annotator confidence in label (0.0 to 1.0)"
        },
        "notes": {
            "type": "string",
            "description": "Additional domain observations or edge cases"
        }
    }
}


@dataclass
class VideoAnnotation:
    """Strongly typed annotation container for an individual clip."""
    video_id: str
    file_name: str
    action_label: str
    camera_viewpoint: str = "unknown"
    is_egocentric: bool = False
    start_sec: Optional[float] = None
    end_sec: Optional[float] = None
    watermark_detected: bool = False
    tool_used: Optional[str] = "none"
    target_surface: Optional[str] = "unknown"
    verified_by_human: bool = False
    annotator_id: Optional[str] = None
    confidence: float = 1.0
    notes: str = ""

    def validate(self, video_duration_sec: Optional[float] = None) -> Tuple[bool, List[str]]:
        """
        Validate this annotation against schema constraints.
        Returns (is_valid, list_of_errors).
        """
        errors = []

        # 1. Action label validation
        if not self.verified_by_human and self.action_label != "unverified":
            errors.append(
                f"Action label '{self.action_label}' cannot be asserted without human verification (verified_by_human is False)."
            )

        if self.action_label not in CANDIDATE_ACTION_CLASSES and self.action_label != "unverified":
            errors.append(
                f"Invalid action_label '{self.action_label}'. Must be one of {sorted(CANDIDATE_ACTION_CLASSES)} or 'unverified'."
            )

        # 2. Camera viewpoint validation
        if self.camera_viewpoint not in CAMERA_VIEWPOINTS:
            errors.append(
                f"Invalid camera_viewpoint '{self.camera_viewpoint}'. Must be one of {sorted(CAMERA_VIEWPOINTS)}."
            )

        if self.is_egocentric and self.camera_viewpoint != "egocentric":
            errors.append(
                f"Conflict: is_egocentric is True, but camera_viewpoint is '{self.camera_viewpoint}'."
            )

        # 3. Temporal timestamp bounds validation
        if self.start_sec is not None and self.start_sec < 0:
            errors.append(f"start_sec cannot be negative (got {self.start_sec}).")

        if self.start_sec is not None and self.end_sec is not None:
            if self.end_sec <= self.start_sec:
                errors.append(
                    f"end_sec ({self.end_sec}) must be strictly greater than start_sec ({self.start_sec})."
                )

        if video_duration_sec is not None and video_duration_sec > 0:
            if self.end_sec is not None and self.end_sec > (video_duration_sec + 0.1):
                errors.append(
                    f"end_sec ({self.end_sec}s) exceeds video duration ({video_duration_sec:.2f}s)."
                )

        # 4. Confidence validation
        if not (0.0 <= self.confidence <= 1.0):
            errors.append(f"confidence must be between 0.0 and 1.0 (got {self.confidence}).")

        return len(errors) == 0, errors

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VideoAnnotation":
        return cls(
            video_id=str(data["video_id"]),
            file_name=str(data["file_name"]),
            action_label=str(data["action_label"]),
            camera_viewpoint=str(data.get("camera_viewpoint", "unknown")),
            is_egocentric=bool(data.get("is_egocentric", False)),
            start_sec=float(data["start_sec"]) if data.get("start_sec") is not None else None,
            end_sec=float(data["end_sec"]) if data.get("end_sec") is not None else None,
            watermark_detected=bool(data.get("watermark_detected", False)),
            tool_used=data.get("tool_used", "none"),
            target_surface=data.get("target_surface", "unknown"),
            verified_by_human=bool(data.get("verified_by_human", False)),
            annotator_id=data.get("annotator_id"),
            confidence=float(data.get("confidence", 1.0)),
            notes=str(data.get("notes", "")),
        )


def export_json_schema(out_path: Path) -> None:
    """Export the formal JSON Schema specification file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ANNOTATION_JSON_SCHEMA, f, indent=2)
