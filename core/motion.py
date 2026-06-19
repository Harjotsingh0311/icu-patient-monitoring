"""
core/motion.py
===============
Frame-to-frame motion energy and facial asymmetry utilities.
"""

from __future__ import annotations
import numpy as np
from typing import Optional

from landmarks.landmark_extractor import LandmarkArray, region_to_array
from landmarks.landmark_groups import (
    JAW_LOWER, LEFT_EYE, RIGHT_EYE,
    LEFT_BROW, RIGHT_BROW,
    LEFT_MOUTH_IDX, RIGHT_MOUTH_IDX, NOSE_TIP_IDX,
)


def compute_motion_energy(
    current: LandmarkArray,
    previous: Optional[LandmarkArray],
    scale: float = 1.0,
) -> float:
    """
    Mean per-landmark Euclidean displacement between consecutive frames.
    Uses jaw and eye landmarks — anatomically meaningful movement regions.
    Returns value normalised to [0, 1] (scale factor converts pixel units).
    """
    if previous is None or len(current) == 0:
        return 0.0

    indices = JAW_LOWER + LEFT_EYE + RIGHT_EYE
    cur_arr  = np.array([[current[i].x,  current[i].y]  for i in indices], dtype=np.float32)
    prev_arr = np.array([[previous[i].x, previous[i].y] for i in indices], dtype=np.float32)

    displacements = np.linalg.norm(cur_arr - prev_arr, axis=1)
    mean_disp = float(np.mean(displacements))

    # Normalise: typical inter-frame motion is 0–5 px; clip at 20 px
    return float(np.clip(mean_disp / 20.0, 0.0, 1.0))


def compute_facial_asymmetry(landmarks: LandmarkArray) -> float:
    """
    Measure of left-right facial asymmetry, normalised to [0, 1].

    Computes unsigned difference in eye and mouth corner distances from
    the midline (nose tip), averaged across feature pairs.
    Large asymmetry may indicate contralateral muscle weakness or
    unilateral pain expression.
    """
    nose = landmarks[NOSE_TIP_IDX]
    le = landmarks[LEFT_EYE[0]]
    re = landmarks[RIGHT_EYE[0]]
    lm_pt = landmarks[LEFT_MOUTH_IDX]
    rm_pt = landmarks[RIGHT_MOUTH_IDX]

    from landmarks.landmark_extractor import dist
    iod = dist(le, re)
    if iod < 1.0:
        return 0.0

    eye_l = abs(nose.x - le.x)
    eye_r = abs(re.x - nose.x)
    mouth_l = abs(nose.x - lm_pt.x)
    mouth_r = abs(rm_pt.x - nose.x)

    eye_asym   = abs(eye_l   - eye_r)   / iod
    mouth_asym = abs(mouth_l - mouth_r) / iod

    return float(np.clip((eye_asym + mouth_asym) / 2.0, 0.0, 1.0))


def compute_perclos(eye_closed_history: list) -> float:
    """
    PERCLOS: Percentage of Eye Closure over recent window.
    eye_closed_history: list/deque of bool or int (1=closed, 0=open)
    Returns fraction ∈ [0, 1].
    """
    if not eye_closed_history:
        return 0.0
    return float(sum(eye_closed_history) / len(eye_closed_history))
