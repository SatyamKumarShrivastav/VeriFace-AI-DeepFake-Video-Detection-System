"""
video_processor.py — Frame extraction and preprocessing for DeepFake detection.
Handles reading video files, sampling frames, and computing per-frame features.
"""

import cv2
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger("deepfake.video_processor")


def extract_frames(video_path: str, max_frames: int = 40, sample_rate: int = 5) -> list[np.ndarray]:
    """
    Extract up to `max_frames` frames from the video at every `sample_rate` frame interval.
    Returns a list of BGR numpy arrays.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    frames = []
    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    # Adaptive sampling to cover the whole video
    if total_frames > 0 and sample_rate * max_frames > total_frames:
        sample_rate = max(1, total_frames // max_frames)

    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_rate == 0:
            frames.append(frame)
        frame_idx += 1

    cap.release()
    logger.info(f"Extracted {len(frames)} frames from {Path(video_path).name} (total={total_frames}, fps={fps:.1f})")
    return frames


def get_video_metadata(video_path: str) -> dict:
    """Return basic metadata about the video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}

    metadata = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "codec": int(cap.get(cv2.CAP_PROP_FOURCC)),
        "duration_sec": 0.0,
    }
    if metadata["fps"] and metadata["fps"] > 0:
        metadata["duration_sec"] = round(metadata["total_frames"] / metadata["fps"], 2)

    cap.release()
    return metadata


def detect_faces_in_frame(frame: np.ndarray) -> list[tuple]:
    """
    Detect face bounding boxes in a BGR frame using the built-in Haar cascade.
    Returns list of (x, y, w, h) tuples.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    return [tuple(f) for f in faces] if len(faces) > 0 else []


def crop_face_region(frame: np.ndarray, bbox: tuple, padding: float = 0.2) -> np.ndarray:
    """Crop a face region from a frame, expanding by `padding` fraction."""
    x, y, w, h = bbox
    ph, pw = int(h * padding), int(w * padding)
    fh, fw = frame.shape[:2]
    x1 = max(0, x - pw)
    y1 = max(0, y - ph)
    x2 = min(fw, x + w + pw)
    y2 = min(fh, y + h + ph)
    return frame[y1:y2, x1:x2]
