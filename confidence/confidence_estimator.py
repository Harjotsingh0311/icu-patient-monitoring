"""
confidence/confidence_estimator.py
====================================
Composite confidence score [0, 1] indicating how much the system trusts
the current distress estimate.

A low confidence score means the system is still producing a distress number,
but the clinician should treat it with scepticism.

Factors
-------
1. Landmark quality  – from z-depth variance of the landmark cloud
2. Head pose         – extreme yaw/pitch/roll partially occlude the face
3. Tracking stability – newly acquired track vs. confirmed patient
4. Face visibility   – face area as fraction of expected (filters tiny/far faces)
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from config.settings import CFG
from landmarks.landmark_extractor import LandmarkArray, estimate_landmark_quality


@dataclass
class ConfidenceResult:
    """Breakdown of confidence components and overall score."""
    overall:           float = 1.0
    landmark_quality:  float = 1.0
    pose_quality:      float = 1.0
    tracking_quality:  float = 1.0
    visibility_quality:float = 1.0

    def as_dict(self) -> dict:
        return {k: round(v, 3) for k, v in self.__dict__.items()}


class ConfidenceEstimator:
    """
    Stateful confidence estimator.  Call `update()` each frame.
    """

    def __init__(self) -> None:
        self._cfg = CFG.confidence
        self._stability_history: list[float] = []

    def update(
        self,
        landmarks: LandmarkArray,
        yaw_deg: float,
        pitch_deg: float,
        roll_deg: float,
        patient_confirmed: bool,
        face_area_fraction: float,   # face bbox area / frame area
    ) -> ConfidenceResult:
        """
        Compute confidence for the current frame.

        Parameters
        ----------
        landmarks           : LandmarkArray  – 468-point MediaPipe landmarks
        yaw_deg/pitch_deg/roll_deg : head pose angles in degrees
        patient_confirmed   : bool – True once tracking is locked
        face_area_fraction  : float – face box area / frame area ∈ [0, 1]

        Returns
        -------
        ConfidenceResult
        """
        cfg = self._cfg

        # ── 1. Landmark quality ──────────────────────────────────────────
        lm_quality = estimate_landmark_quality(landmarks)

        # ── 2. Head pose ─────────────────────────────────────────────────
        yaw_ok   = max(0.0, 1.0 - abs(yaw_deg)   / cfg.max_yaw_deg)
        pitch_ok = max(0.0, 1.0 - abs(pitch_deg) / cfg.max_pitch_deg)
        roll_ok  = max(0.0, 1.0 - abs(roll_deg)  / cfg.max_roll_deg)
        pose_quality = float((yaw_ok + pitch_ok + roll_ok) / 3.0)

        # ── 3. Tracking stability ────────────────────────────────────────
        # Ramp from 0→1 over `tracking_stability_window` consecutive confirmed frames
        if patient_confirmed:
            track_q = 1.0
        else:
            n = len(self._stability_history)
            track_q = min(n / max(cfg.tracking_stability_window, 1), 1.0)
        self._stability_history.append(float(patient_confirmed))
        if len(self._stability_history) > cfg.tracking_stability_window * 2:
            self._stability_history = self._stability_history[-cfg.tracking_stability_window:]

        # ── 4. Face visibility (size) ────────────────────────────────────
        # Expect face to occupy ~5-40% of frame; very small = far / occluded
        vis_quality = float(np.clip(face_area_fraction / 0.05, 0.0, 1.0))

        # ── Overall (geometric mean to penalise any single bad factor) ───
        overall = float(
            (lm_quality ** 0.30) *
            (pose_quality ** 0.30) *
            (track_q ** 0.25) *
            (vis_quality ** 0.15)
        )
        overall = float(np.clip(overall, 0.0, 1.0))

        return ConfidenceResult(
            overall=overall,
            landmark_quality=lm_quality,
            pose_quality=pose_quality,
            tracking_quality=track_q,
            visibility_quality=vis_quality,
        )
