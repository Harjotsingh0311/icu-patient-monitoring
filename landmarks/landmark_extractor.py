"""
landmarks/landmark_extractor.py
================================
Converts raw MediaPipe face-mesh output into typed numpy arrays
used throughout the AU estimation pipeline.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class Landmark:
    """Single 3-D facial landmark in pixel coordinates."""
    x: float
    y: float
    z: float          # relative depth (negative = towards camera)
    visibility: float = 1.0


LandmarkArray = List[Landmark]


def extract_landmarks(
    face_landmarks,
    width: int,
    height: int,
    visibility_threshold: float = 0.0,
) -> LandmarkArray:
    """
    Convert a MediaPipe NormalizedLandmarkList to a list of Landmark objects.

    Parameters
    ----------
    face_landmarks : mediapipe NormalizedLandmarkList
    width, height  : frame dimensions in pixels
    visibility_threshold : landmarks below this are flagged (future use)

    Returns
    -------
    List[Landmark]  – indexed 0..467 matching MediaPipe ordering
    """
    landmarks: LandmarkArray = []
    for lm in face_landmarks.landmark:
        landmarks.append(
            Landmark(
                x=lm.x * width,
                y=lm.y * height,
                z=lm.z,
                visibility=getattr(lm, "visibility", 1.0),
            )
        )
    return landmarks


def get_region(landmarks: LandmarkArray, indices: List[int]) -> LandmarkArray:
    """Return a sub-list of landmarks by index."""
    return [landmarks[i] for i in indices]


def landmark_to_array(lm: Landmark) -> np.ndarray:
    """Convert a Landmark to a (2,) float32 pixel array."""
    return np.array([lm.x, lm.y], dtype=np.float32)


def region_to_array(region: LandmarkArray) -> np.ndarray:
    """Convert a list of Landmarks to an (N, 2) float32 array."""
    return np.array([[lm.x, lm.y] for lm in region], dtype=np.float32)


def dist(a: Landmark, b: Landmark) -> float:
    """Euclidean distance between two landmarks (pixels)."""
    return float(np.hypot(a.x - b.x, a.y - b.y))


def interocular_distance(landmarks: LandmarkArray) -> float:
    """
    Distance between outer eye corners.
    Used as a normalisation denominator for all geometric measurements,
    making all ratios scale-invariant across face sizes and camera distances.
    """
    from landmarks.landmark_groups import LEFT_EYE_OUTER_IDX, RIGHT_EYE_OUTER_IDX
    return max(
        dist(landmarks[LEFT_EYE_OUTER_IDX], landmarks[RIGHT_EYE_OUTER_IDX]),
        1.0,   # guard against division-by-zero
    )


def estimate_landmark_quality(landmarks: LandmarkArray) -> float:
    """
    Heuristic landmark quality score ∈ [0, 1].
    Based on z-depth variance (low variance when face is frontal & well-lit).
    """
    zs = np.array([lm.z for lm in landmarks], dtype=np.float32)
    z_std = float(np.std(zs))
    # Empirically: z_std < 0.05 is very frontal; > 0.20 is heavily posed
    quality = float(np.clip(1.0 - z_std / 0.20, 0.0, 1.0))
    return quality
