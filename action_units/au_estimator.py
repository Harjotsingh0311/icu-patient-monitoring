"""
action_units/au_estimator.py
============================
Geometric estimation of 15 FACS Action Units from MediaPipe 468 landmarks.

Clinical rationale
------------------
Action Units (AUs) are the atomic muscle-group activations defined in
Ekman's Facial Action Coding System (FACS).  Pain expression research
(Prkachin & Solomon, 2008) identifies AU4, AU6, AU7, AU9, AU10, and AU43
as the most diagnostically significant for acute pain.  All AUs estimated
here use normalised geometric ratios derived from interocular distance,
making them invariant to face size and camera zoom.

Each AU intensity is in [0.0, 1.0]:
  0.0 = AU not present / baseline
  1.0 = maximum estimated activation
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from landmarks.landmark_extractor import (
    LandmarkArray, dist, interocular_distance, landmark_to_array
)
from landmarks.landmark_groups import (
    LEFT_EYE, RIGHT_EYE,
    LEFT_BROW, RIGHT_BROW,
    LEFT_BROW_INNER, RIGHT_BROW_INNER,
    LEFT_CHEEK, RIGHT_CHEEK,
    LEFT_NOSTRIL, RIGHT_NOSTRIL,
    UPPER_LIP_CENTER, LOWER_LIP_CENTER,
    MOUTH_LEFT_CORNER, MOUTH_RIGHT_CORNER,
    UPPER_LIP_INNER, LOWER_LIP_INNER,
    JAW_LOWER, CHIN,
    NOSE_TIP_IDX, CHIN_IDX,
    LEFT_MOUTH_IDX, RIGHT_MOUTH_IDX,
)
from config.settings import CFG


@dataclass
class AUFrame:
    """Snapshot of all estimated AU intensities for a single frame."""
    au4:  float = 0.0   # Brow Lowerer
    au5:  float = 0.0   # Upper Lid Raiser
    au6:  float = 0.0   # Cheek Raiser
    au7:  float = 0.0   # Lid Tightener
    au9:  float = 0.0   # Nose Wrinkler
    au10: float = 0.0   # Upper Lip Raiser
    au12: float = 0.0   # Lip Corner Puller (smile)
    au15: float = 0.0   # Lip Corner Depressor
    au17: float = 0.0   # Chin Raiser
    au20: float = 0.0   # Lip Stretcher
    au23: float = 0.0   # Lip Tightener
    au24: float = 0.0   # Lip Pressor
    au25: float = 0.0   # Lips Part
    au26: float = 0.0   # Jaw Drop
    au43: float = 0.0   # Eyes Closed

    # Derived convenience values (not proper AUs)
    ear_left:  float = 0.0
    ear_right: float = 0.0
    ear_avg:   float = 0.0
    mar:       float = 0.0
    brow_distance_ratio: float = 0.0  # normalised brow-to-eye distance

    def as_dict(self) -> dict:
        return {k: round(v, 4) for k, v in self.__dict__.items()}


def _scale(value: float, low: float, high: float) -> float:
    """Linearly map value from [low, high] → [0, 1], clamped."""
    if high <= low:
        return 0.0
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def _ear(eye: LandmarkArray) -> float:
    """
    Eye Aspect Ratio (Soukupová & Čech, 2016).
    eye must have 6 landmarks: p0 outer, p1 upper-outer, p2 upper-inner,
                                p3 inner, p4 lower-inner, p5 lower-outer.
    """
    p0, p1, p2, p3, p4, p5 = eye
    v1 = dist(p1, p5)
    v2 = dist(p2, p4)
    h  = dist(p0, p3)
    if h < 1e-6:
        return 0.0
    return (v1 + v2) / (2.0 * h)


def estimate_aus(
    landmarks: LandmarkArray,
    iod: float | None = None,
) -> AUFrame:
    """
    Compute all 15 Action Unit intensities from a landmark array.

    Parameters
    ----------
    landmarks : LandmarkArray  – 468-point MediaPipe output
    iod       : float          – interocular distance in px (computed if None)

    Returns
    -------
    AUFrame
    """
    cfg = CFG.au
    if iod is None:
        iod = interocular_distance(landmarks)
    iod = max(iod, 1.0)

    au = AUFrame()

    # ── convenience helper ──────────────────────────────────────────────
    def _lm(idx: int) -> "Landmark":
        return landmarks[idx]

    def _region(indices) -> LandmarkArray:
        return [landmarks[i] for i in indices]

    # ── EAR (used by multiple AUs) ───────────────────────────────────────
    left_eye  = _region(LEFT_EYE)
    right_eye = _region(RIGHT_EYE)
    ear_l = _ear(left_eye)
    ear_r = _ear(right_eye)
    ear_avg = (ear_l + ear_r) / 2.0

    au.ear_left  = ear_l
    au.ear_right = ear_r
    au.ear_avg   = ear_avg

    # ────────────────────────────────────────────────────────────────────
    # AU43  Eyes Closed  (inverse of EAR)
    # Clinical: sustained eye closure signals pain-driven withdrawal or fatigue.
    # ────────────────────────────────────────────────────────────────────
    au.au43 = _scale(cfg.au43_high - ear_avg, 0.0, cfg.au43_high - cfg.au43_low)

    # ────────────────────────────────────────────────────────────────────
    # AU5   Upper Lid Raiser  (high EAR = wide open eyes = fear/shock)
    # ────────────────────────────────────────────────────────────────────
    au.au5 = _scale(ear_avg, cfg.au5_low, cfg.au5_high)

    # ────────────────────────────────────────────────────────────────────
    # AU7   Lid Tightener  (slight squint: moderate EAR reduction)
    # Clinical: tightening the upper lid creates a "squinting" expression
    # characteristic of pain and concentration.
    # ────────────────────────────────────────────────────────────────────
    # Lid tightening = deviation below a mid-open EAR (0.25) but above closure
    mid_ear = (cfg.au43_low + cfg.au5_low) / 2.0
    au7_raw = max(0.0, mid_ear - ear_avg)          # positive when below mid
    au.au7 = _scale(au7_raw, cfg.au7_low, cfg.au7_high)

    # ────────────────────────────────────────────────────────────────────
    # AU4   Brow Lowerer
    # Measured as brow-to-upper-eyelid distance, normalised by IOD.
    # Lower ratio → brows pressed downward.
    # Clinical: the single most reliable pain indicator (Prkachin 1992).
    # ────────────────────────────────────────────────────────────────────
    left_brow  = _region(LEFT_BROW)
    right_brow = _region(RIGHT_BROW)

    # mid-brow y vs upper eyelid y (MediaPipe y=0 is top of frame)
    left_brow_y  = float(np.mean([lm.y for lm in left_brow]))
    right_brow_y = float(np.mean([lm.y for lm in right_brow]))
    left_eye_top_y  = left_eye[1].y   # upper lid outer
    right_eye_top_y = right_eye[1].y

    left_brow_dist  = (left_eye_top_y  - left_brow_y)  / iod
    right_brow_dist = (right_eye_top_y - right_brow_y) / iod
    brow_dist_avg   = (left_brow_dist + right_brow_dist) / 2.0

    au.brow_distance_ratio = brow_dist_avg
    # AU4 increases as brow-eye gap SHRINKS below normal (invert the scale)
    au.au4 = _scale(cfg.au4_high - brow_dist_avg, 0.0, cfg.au4_high - cfg.au4_low)

    # ────────────────────────────────────────────────────────────────────
    # AU6   Cheek Raiser
    # Cheek bulk rises when the zygomatic major and orbicularis oculi activate,
    # narrowing the palpebral fissure from below.  Estimated as reduction in
    # the lower eyelid–cheek distance normalised by IOD.
    # Clinical: distinguishes genuine pain grimace from AU4 alone.
    # ────────────────────────────────────────────────────────────────────
    left_cheek  = _region(LEFT_CHEEK)
    right_cheek = _region(RIGHT_CHEEK)
    left_cheek_y  = float(np.mean([lm.y for lm in left_cheek]))
    right_cheek_y = float(np.mean([lm.y for lm in right_cheek]))
    left_lower_lid_y  = left_eye[4].y    # lower lid inner
    right_lower_lid_y = right_eye[4].y

    left_cheek_gap  = (left_cheek_y  - left_lower_lid_y)  / iod
    right_cheek_gap = (right_cheek_y - right_lower_lid_y) / iod
    cheek_gap_avg   = (left_cheek_gap + right_cheek_gap) / 2.0

    # Smaller gap = cheeks more raised
    au.au6 = _scale(cfg.au6_high - cheek_gap_avg, 0.0, cfg.au6_high - cfg.au6_low)

    # ────────────────────────────────────────────────────────────────────
    # AU9   Nose Wrinkler
    # Estimated as the relative width of the nostrils divided by nose bridge
    # width — nasal flare and wrinkling widens the nostrils.
    # Clinical: reliably accompanies disgust and pain expressions.
    # ────────────────────────────────────────────────────────────────────
    left_nos  = _region(LEFT_NOSTRIL)
    right_nos = _region(RIGHT_NOSTRIL)
    nostril_left_x  = float(np.mean([lm.x for lm in left_nos]))
    nostril_right_x = float(np.mean([lm.x for lm in right_nos]))
    nostril_width   = abs(nostril_right_x - nostril_left_x) / iod

    # Reference: internostril width at rest ≈ 0.20–0.25 × IOD
    NOSTRIL_NEUTRAL = 0.22
    au9_raw = max(0.0, nostril_width - NOSTRIL_NEUTRAL)
    au.au9 = _scale(au9_raw, cfg.au9_low, cfg.au9_high)

    # ────────────────────────────────────────────────────────────────────
    # AU10  Upper Lip Raiser
    # Distance from nose base to upper lip, normalised.  Decreases when
    # the upper lip is raised, shortening the philtrum.
    # ────────────────────────────────────────────────────────────────────
    nose_tip   = _lm(NOSE_TIP_IDX)
    upper_lip  = _lm(UPPER_LIP_CENTER)
    philtrum   = dist(nose_tip, upper_lip) / iod

    PHILTRUM_NEUTRAL = 0.28
    au10_raw = max(0.0, PHILTRUM_NEUTRAL - philtrum)   # shorter = raised
    au.au10 = _scale(au10_raw, cfg.au10_low, cfg.au10_high)

    # ────────────────────────────────────────────────────────────────────
    # AU12  Lip Corner Puller (smile / grimace pull)
    # Horizontal displacement of mouth corners relative to neutral.
    # ────────────────────────────────────────────────────────────────────
    l_corner = _lm(MOUTH_LEFT_CORNER)
    r_corner = _lm(MOUTH_RIGHT_CORNER)
    mouth_width = dist(l_corner, r_corner) / iod

    MOUTH_WIDTH_NEUTRAL = 0.45
    au12_raw = max(0.0, mouth_width - MOUTH_WIDTH_NEUTRAL)
    au.au12 = _scale(au12_raw, cfg.au12_low, cfg.au12_high)

    # ────────────────────────────────────────────────────────────────────
    # AU15  Lip Corner Depressor
    # Downward displacement of mouth corners.  The depressor anguli oris
    # pulls corners down, a hallmark of sadness and pain.
    # ────────────────────────────────────────────────────────────────────
    lower_lip_y  = _lm(LOWER_LIP_CENTER).y
    l_corner_y   = l_corner.y
    r_corner_y   = r_corner.y
    corner_vs_lip_avg = ((l_corner_y + r_corner_y) / 2.0 - lower_lip_y) / iod
    au15_raw = max(0.0, corner_vs_lip_avg)
    au.au15 = _scale(au15_raw, cfg.au15_low, cfg.au15_high)

    # ────────────────────────────────────────────────────────────────────
    # AU17  Chin Raiser
    # Mentalis muscle raises chin pad and pushes lower lip upward.
    # Measured as reduction in chin-to-lip distance.
    # ────────────────────────────────────────────────────────────────────
    chin_lm = _lm(CHIN_IDX)
    lower_lip_lm = _lm(LOWER_LIP_CENTER)
    chin_lip_dist = dist(chin_lm, lower_lip_lm) / iod

    CHIN_LIP_NEUTRAL = 0.18
    au17_raw = max(0.0, CHIN_LIP_NEUTRAL - chin_lip_dist)
    au.au17 = _scale(au17_raw, cfg.au17_low, cfg.au17_high)

    # ────────────────────────────────────────────────────────────────────
    # AU20  Lip Stretcher
    # Risorius / platysma stretch lips horizontally.
    # Uses same mouth width as AU12 but captures a different range.
    # ────────────────────────────────────────────────────────────────────
    lip_stretch = mouth_width   # reuse from AU12
    au.au20 = _scale(lip_stretch, cfg.au20_low, cfg.au20_high)

    # ────────────────────────────────────────────────────────────────────
    # AU23  Lip Tightener
    # Obicularis oris compresses the lips.  Measured as inner lip thickness
    # reduction.
    # ────────────────────────────────────────────────────────────────────
    upper_inner = _region(UPPER_LIP_INNER)
    lower_inner = _region(LOWER_LIP_INNER)
    upper_inner_y = float(np.mean([lm.y for lm in upper_inner]))
    lower_inner_y = float(np.mean([lm.y for lm in lower_inner]))
    lip_thickness = abs(lower_inner_y - upper_inner_y) / iod

    LIP_THICKNESS_NEUTRAL = 0.08
    au23_raw = max(0.0, LIP_THICKNESS_NEUTRAL - lip_thickness)
    au.au23 = _scale(au23_raw, cfg.au23_low, cfg.au23_high)

    # ────────────────────────────────────────────────────────────────────
    # AU24  Lip Pressor
    # Upper and lower lips pressed together — inter-lip gap approaches 0.
    # Distinct from AU23: AU24 is pressing shut, AU23 is thinning.
    # ────────────────────────────────────────────────────────────────────
    inter_lip = abs(_lm(UPPER_LIP_CENTER).y - _lm(LOWER_LIP_CENTER).y) / iod
    # Very small inter_lip = pressed lips
    au24_raw = max(0.0, cfg.au24_high - inter_lip)
    au.au24 = _scale(au24_raw, 0.0, cfg.au24_high)

    # ────────────────────────────────────────────────────────────────────
    # AU25  Lips Part
    # Simple inter-lip distance — increases with mouth opening.
    # ────────────────────────────────────────────────────────────────────
    au.au25 = _scale(inter_lip, cfg.au25_low, cfg.au25_high)

    # MAR (Mouth Aspect Ratio) stored for downstream use
    if mouth_width * iod > 1e-3:
        au.mar = (inter_lip * iod) / (mouth_width * iod)
    else:
        au.mar = 0.0

    # ────────────────────────────────────────────────────────────────────
    # AU26  Jaw Drop
    # Masseter relaxation allows jaw to drop.  Estimated as chin-to-nose
    # distance increase.
    # ────────────────────────────────────────────────────────────────────
    face_height = dist(_lm(NOSE_TIP_IDX), _lm(CHIN_IDX)) / iod
    FACE_HEIGHT_NEUTRAL = 0.90
    jaw_drop_raw = max(0.0, face_height - FACE_HEIGHT_NEUTRAL)
    au.au26 = _scale(jaw_drop_raw, cfg.au26_low, cfg.au26_high)

    return au
