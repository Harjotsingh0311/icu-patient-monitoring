"""
tracking/patient_tracker.py
============================
Maintains patient identity across frames when multiple faces are present
(nurses, doctors, visitors may enter the camera field).

Strategy
--------
1. On first detection, the largest / most-central face is presumed the patient.
2. Each subsequent frame, faces are matched to the patient using IoU overlap
   of their bounding boxes.
3. If the patient face is lost for `max_lost_frames`, the tracker resets and
   re-acquires — never accidentally scoring a different face long-term.

Only the tracked patient face is passed to the AU estimator and scorer.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np

from config.settings import CFG


@dataclass
class FaceBox:
    """Axis-aligned bounding box for a detected face."""
    x1: float
    y1: float
    x2: float
    y2: float
    face_index: int    # index into mediapipe's multi_face_landmarks list

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0


def _iou(a: FaceBox, b: FaceBox) -> float:
    """Intersection-over-Union of two bounding boxes."""
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def landmarks_to_box(
    face_landmarks,
    width: int,
    height: int,
    face_index: int,
) -> FaceBox:
    """Compute a bounding box from a MediaPipe NormalizedLandmarkList."""
    xs = [lm.x * width  for lm in face_landmarks.landmark]
    ys = [lm.y * height for lm in face_landmarks.landmark]
    return FaceBox(
        x1=float(min(xs)),
        y1=float(min(ys)),
        x2=float(max(xs)),
        y2=float(max(ys)),
        face_index=face_index,
    )


class PatientTracker:
    """
    Tracks a single patient face across frames.

    Usage
    -----
    tracker = PatientTracker(frame_width, frame_height)
    for each frame:
        boxes = [landmarks_to_box(fl, w, h, i)
                 for i, fl in enumerate(results.multi_face_landmarks)]
        patient_box = tracker.update(boxes)
        if patient_box is not None:
            patient_landmarks = results.multi_face_landmarks[patient_box.face_index]
    """

    def __init__(self, frame_width: int, frame_height: int) -> None:
        self._cfg = CFG.tracking
        self._w = frame_width
        self._h = frame_height
        self._patient_box: Optional[FaceBox] = None
        self._lost_frames: int = 0
        self._lock_frames: int = 0
        self.patient_confirmed: bool = False

    def update(self, boxes: List[FaceBox]) -> Optional[FaceBox]:
        """
        Match incoming face boxes to the tracked patient.

        Returns
        -------
        FaceBox | None  – the patient box if successfully tracked, else None
        """
        if not boxes:
            return self._handle_no_faces()

        if self._patient_box is None:
            # Acquire: pick the largest face (bed patient typically closest)
            self._patient_box = max(boxes, key=lambda b: b.area)
            self._lost_frames = 0
            self._lock_frames = 0
            self.patient_confirmed = False
            return self._patient_box

        # Match to existing patient track by highest IoU
        best_box: Optional[FaceBox] = None
        best_iou: float = 0.0
        for box in boxes:
            iou = _iou(self._patient_box, box)
            if iou > best_iou:
                best_iou = iou
                best_box = box

        if best_iou >= self._cfg.iou_threshold and best_box is not None:
            self._patient_box = best_box
            self._lost_frames = 0
            self._lock_frames = min(
                self._lock_frames + 1,
                self._cfg.patient_lock_frames
            )
            self.patient_confirmed = (
                self._lock_frames >= self._cfg.patient_lock_frames
            )
            return self._patient_box
        else:
            return self._handle_no_faces()

    def _handle_no_faces(self) -> Optional[FaceBox]:
        self._lost_frames += 1
        if self._lost_frames >= self._cfg.max_lost_frames:
            # Reset tracking
            self._patient_box = None
            self._lost_frames = 0
            self._lock_frames = 0
            self.patient_confirmed = False
        return None

    @property
    def patient_box(self) -> Optional[FaceBox]:
        return self._patient_box

    def draw_box(self, frame, color: tuple = (0, 200, 80)) -> None:
        """Draw patient bounding box onto frame (BGR)."""
        import cv2
        if self._patient_box is None:
            return
        b = self._patient_box
        cv2.rectangle(
            frame,
            (int(b.x1), int(b.y1)),
            (int(b.x2), int(b.y2)),
            color, 2
        )
        label = "PATIENT" if self.patient_confirmed else "ACQUIRING..."
        cv2.putText(
            frame, label,
            (int(b.x1), int(b.y1) - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1
        )
