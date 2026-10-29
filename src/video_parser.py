"""
Video metadata extraction and validation module.
Supports MP4, MOV, AVI, and MKV formats with pure-Python container inspection
and optional OpenCV acceleration.
"""
from dataclasses import dataclass, asdict
import hashlib
import os
from pathlib import Path
import struct
from typing import Optional, Tuple, Dict, Any


@dataclass
class VideoMetadata:
    """Metadata container for a single video file."""
    file_path: str
    file_name: str
    file_size_bytes: int
    format: str
    sha256: str
    duration_sec: Optional[float] = None
    frame_count: Optional[int] = None
    fps: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    is_readable: bool = True
    validation_status: str = "valid"
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_sha256(file_path: Path, block_size: int = 65536) -> str:
    """Compute SHA-256 hash of a file efficiently."""
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            sha.update(block)
    return sha.hexdigest()


def parse_mp4_mov(file_path: Path) -> Tuple[Optional[float], Optional[int], Optional[int], Optional[int]]:
    """
    Parse ISO base media file format (MP4/MOV) atoms:
    Reads mvhd for duration, stsz for frame count, and visual sample description for dimensions.
    """
    duration = None
    frame_count = None
    width, height = None, None
    timescale = 1

    try:
        with open(file_path, "rb") as f:
            file_size = os.fstat(f.fileno()).st_size
            
            # Helper to recursively step through container boxes
            def walk_boxes(start: int, end: int, depth: int = 0):
                nonlocal duration, frame_count, width, height, timescale
                if depth > 10:  # Prevent recursion cycles
                    return
                f.seek(start)
                while f.tell() < end:
                    header = f.read(8)
                    if len(header) < 8:
                        break
                    size, btype = struct.unpack(">I4s", header)
                    if size == 1:
                        large_size = f.read(8)
                        if len(large_size) < 8:
                            break
                        size = struct.unpack(">Q", large_size)[0]
                        box_end = f.tell() - 16 + size
                    elif size == 0:
                        box_end = end
                    else:
                        box_end = f.tell() - 8 + size

                    if box_end > end or box_end < f.tell():
                        box_end = end

                    if btype in (b"moov", b"trak", b"mdia", b"minf", b"stbl"):
                        walk_boxes(f.tell(), box_end, depth + 1)
                    elif btype == b"mvhd":
                        v_data = f.read(4)
                        if len(v_data) == 4:
                            v = v_data[0]
                            if v == 0:
                                d = f.read(16)
                                if len(d) == 16:
                                    _, _, ts, dur = struct.unpack(">IIII", d)
                                    if ts > 0:
                                        timescale = ts
                                        duration = dur / ts
                            elif v == 1:
                                d = f.read(24)
                                if len(d) == 24:
                                    _, _, ts, dur = struct.unpack(">QQIQ", d[:28]) if len(d) >= 28 else (0, 0, 1, 0)
                                    if ts > 0:
                                        timescale = ts
                                        duration = dur / ts
                    elif btype == b"stsz" and frame_count is None:
                        v_data = f.read(4)
                        s_data = f.read(8)
                        if len(s_data) == 8:
                            _, sample_count = struct.unpack(">II", s_data)
                            frame_count = sample_count

                    f.seek(box_end)

            walk_boxes(0, file_size)

            # Extract resolution from visual sample description or moov data
            f.seek(0)
            prefix = f.read(min(32768, file_size))
            # Scan for known visual sample descriptors: avc1, mp4v, hvc1, hev1
            for codec in [b"avc1", b"mp4v", b"hvc1", b"hev1"]:
                idx = prefix.rfind(codec)
                if idx != -1 and idx + 32 <= len(prefix):
                    w, h = struct.unpack(">HH", prefix[idx + 28 : idx + 32])
                    if w > 0 and h > 0:
                        width, height = w, h
                        break

    except Exception:
        pass

    return duration, frame_count, width, height


def parse_avi(file_path: Path) -> Tuple[Optional[float], Optional[int], Optional[int], Optional[int]]:
    """Parse RIFF AVI header to extract duration, frames, and dimensions."""
    duration = None
    frame_count = None
    width, height = None, None

    try:
        with open(file_path, "rb") as f:
            header = f.read(128)
            if len(header) >= 64 and header[:4] == b"RIFF" and header[8:12] == b"AVI ":
                # Find avih chunk
                idx = header.find(b"avih")
                if idx != -1 and idx + 44 <= len(header):
                    # avih structure: 4s id, I size, I microsec_per_frame, I max_bytes, I padding, I flags, I total_frames, ...
                    data = header[idx + 8 : idx + 48]
                    microsec, _, _, _, total_frames = struct.unpack("<IIIII", data[:20])
                    w, h = struct.unpack("<II", data[32:40])
                    if total_frames > 0:
                        frame_count = total_frames
                    if microsec > 0 and frame_count:
                        duration = (microsec * frame_count) / 1_000_000.0
                    if w > 0 and h > 0:
                        width, height = w, h
    except Exception:
        pass

    return duration, frame_count, width, height


def parse_mkv(file_path: Path) -> Tuple[Optional[float], Optional[int], Optional[int], Optional[int]]:
    """Basic parser for Matroska (MKV) container verifying EBML header."""
    duration = None
    frame_count = None
    width, height = None, None

    try:
        with open(file_path, "rb") as f:
            header = f.read(64)
            # EBML ID is 0x1A45DFA3
            if len(header) >= 4 and header[:4] == b"\x1a\x45\xdf\xa3":
                # Valid MKV container detected
                # Full EBML parsing can be complemented with cv2 when installed
                pass
    except Exception:
        pass

    return duration, frame_count, width, height


def extract_video_metadata(file_path: Path) -> VideoMetadata:
    """
    Extract comprehensive video metadata and validate integrity.
    Detects empty files, unreadable files, corrupt headers, and extracts dimensions.
    """
    path_obj = Path(file_path).resolve()
    file_name = path_obj.name
    ext = path_obj.suffix.lower()

    if not path_obj.exists():
        return VideoMetadata(
            file_path=str(path_obj),
            file_name=file_name,
            file_size_bytes=0,
            format=ext,
            sha256="",
            is_readable=False,
            validation_status="unreadable",
            error_message="File does not exist",
        )

    file_size = path_obj.stat().st_size
    if file_size == 0:
        return VideoMetadata(
            file_path=str(path_obj),
            file_name=file_name,
            file_size_bytes=0,
            format=ext,
            sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            is_readable=False,
            validation_status="empty",
            error_message="File is 0 bytes (empty video)",
        )

    sha256 = compute_sha256(path_obj)
    duration, frame_count, width, height = None, None, None, None

    # Parse container according to extension
    if ext in (".mp4", ".mov"):
        duration, frame_count, width, height = parse_mp4_mov(path_obj)
    elif ext == ".avi":
        duration, frame_count, width, height = parse_avi(path_obj)
    elif ext == ".mkv":
        duration, frame_count, width, height = parse_mkv(path_obj)

    # Optional OpenCV check if cv2 is installed and some attributes are missing
    try:
        import cv2
        cap = cv2.VideoCapture(str(path_obj))
        if cap.isOpened():
            if duration is None:
                fps_cv = cap.get(cv2.CAP_PROP_FPS)
                frames_cv = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if frames_cv > 0 and fps_cv > 0:
                    duration = frames_cv / fps_cv
                    frame_count = frames_cv
            if width is None or height is None:
                w_cv = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h_cv = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if w_cv > 0 and h_cv > 0:
                    width, height = w_cv, h_cv
            cap.release()
    except ImportError:
        pass
    except Exception:
        pass

    # Validation verdict
    is_readable = True
    status = "valid"
    err = None

    if duration is None and frame_count is None and (width is None or height is None):
        is_readable = False
        status = "corrupt"
        err = f"Failed to parse valid video stream from {ext} container"

    fps = (frame_count / duration) if (duration and frame_count and duration > 0) else None

    return VideoMetadata(
        file_path=str(path_obj),
        file_name=file_name,
        file_size_bytes=file_size,
        format=ext,
        sha256=sha256,
        duration_sec=round(duration, 3) if duration else None,
        frame_count=frame_count,
        fps=round(fps, 2) if fps else None,
        width=width,
        height=height,
        is_readable=is_readable,
        validation_status=status,
        error_message=err,
    )
