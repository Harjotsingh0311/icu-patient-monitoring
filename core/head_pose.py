"""
core/head_pose.py
==================
Estimates head pose (yaw, pitch, roll) from 6 stable facial landmarks
using OpenCV's solvePnP with a canonical 3-D face model.
"""

from __future__ import annotations
import cv2
import numpy as np

from landmarks.landmark_extractor import LandmarkArray


# 3-D reference face model in mm (canonical neutral pose)
_MODEL_POINTS = np.array([
    ( 0.0,    0.0,    0.0),   # Nose tip
    ( 0.0,  -63.6,  -12.5),  # Chin
    (-43.3,  32.7,  -26.0),  # Left eye outer corner
    ( 43.3,  32.7,  -26.0),  # Right eye outer corner
    (-28.9, -28.9,  -24.1),  # Left mouth corner
    ( 28.9, -28.9,  -24.1),  # Right mouth corner
], dtype=np.float64)

# Corresponding MediaPipe landmark indices
_LANDMARK_INDICES = [1, 152, 33, 263, 61, 291]


def estimate_head_pose(
    landmarks: LandmarkArray,
    frame_width: int,
    frame_height: int,
) -> tuple[float, float, float]:
    """
    Estimate head orientation using PnP.

    Returns
    -------
    (yaw, pitch, roll) in degrees
    Positive yaw  = face turned right
    Positive pitch = face tilted upward
    Positive roll  = face tilted right
    """
    image_points = np.array(
        [(landmarks[i].x, landmarks[i].y) for i in _LANDMARK_INDICES],
        dtype=np.float64,
    )

    focal_length = float(frame_width)
    cx = frame_width  / 2.0
    cy = frame_height / 2.0
    camera_matrix = np.array(
        [[focal_length, 0, cx],
         [0, focal_length, cy],
         [0, 0, 1]],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rvec, tvec = cv2.solvePnP(
        _MODEL_POINTS, image_points,
        camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return 0.0, 0.0, 0.0

    rmat, _ = cv2.Rodrigues(rvec)
    angles, *_ = cv2.RQDecomp3x3(rmat)
    pitch, yaw, roll = angles[0], angles[1], angles[2]
    return float(yaw), float(pitch), float(roll)


def head_pose_deviation(
    yaw: float, pitch: float, roll: float
) -> float:
    """
    Scalar deviation of head pose from frontal (0,0,0), normalised to [0,1].
    """
    from config.settings import CFG
    cfg = CFG.confidence
    d = (
        min(abs(yaw)   / cfg.max_yaw_deg,   1.0) +
        min(abs(pitch) / cfg.max_pitch_deg, 1.0) +
        min(abs(roll)  / cfg.max_roll_deg,  1.0)
    ) / 3.0
    return float(np.clip(d, 0.0, 1.0))
