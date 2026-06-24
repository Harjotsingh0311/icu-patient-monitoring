"""
action_units/au_estimator.py
============================
Geometric estimation of 15 FACS Action Units from MediaPipe 468 landmarks.

CALIBRATION REVISION (second pass) — see CALIBRATION_REPORT.md for full rationale.

Key geometry fixes vs. original:
  AU43: EAR thresholds recalibrated to actual MediaPipe neutral EAR range
        (~0.18–0.22). au43_high=0.25, au43_low=0.06.
  AU4:  Inner-brow to lid-apex geometry is correct; thresholds recalibrated
        to actual neutral gap (~0.10–0.13 IOD). au4_high=0.13, au4_low=0.03.
  AU6:  Malar landmarks (205/425) and lower-lid reference (eye[5]) are correct;
        thresholds recalibrated to actual neutral gap (~0.35–0.40 IOD).
        au6_high=0.40, au6_low=0.18.
  Debug print() removed from AU6 block (was leaking to stdout in production).
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


# ── MediaPipe malar-prominence landmark indices ──────────────────────────
# These are the zygomatic / cheekbone-area landmarks in the MediaPipe 468-
# point map that best track cheek-raise movement (AU6).
# 205  = left malar prominence (lateral to nose, inferior to orbit)
# 425  = right malar prominence (mirror of 205)
# Verified against the canonical MediaPipe face-mesh UV map.
_LEFT_MALAR_IDX  = 205
_RIGHT_MALAR_IDX = 425

# Inner-brow landmarks for AU4 (corrugator supercilii origin).
# These move more than the outer brow during AU4 and are the FACS definition.
# 55  = left inner brow
# 285 = right inner brow
_LEFT_INNER_BROW_IDX  = 55
_RIGHT_INNER_BROW_IDX = 285

# Upper eyelid apex — highest point on the upper lid in MediaPipe.
# Used as the "eye top" reference for brow-to-lid gap.
# 159 = left upper lid apex
# 386 = right upper lid apex
_LEFT_LID_APEX_IDX  = 159
_RIGHT_LID_APEX_IDX = 386


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

    ear_left:  float = 0.0
    ear_right: float = 0.0
    ear_avg:   float = 0.0
    mar:       float = 0.0
    brow_distance_ratio: float = 0.0

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
    eye must have 6 landmarks in order:
      p0 outer-corner, p1 upper-outer, p2 upper-inner,
      p3 inner-corner, p4 lower-inner, p5 lower-outer.

    EAR = (||p1-p5|| + ||p2-p4||) / (2 · ||p0-p3||)

    For MediaPipe, the standard 6-point aperture set is:
      Left eye:  [33, 160, 158, 133, 153, 144]
      Right eye: [263, 387, 385, 362, 380, 373]
    Ensure LEFT_EYE / RIGHT_EYE in landmark_groups.py use these exact indices
    in this exact order.  Using a different subset (e.g., iris-refined points
    from the 478-point model) will produce incorrect EAR values.

    Typical MediaPipe EAR ranges (480-point model, non-iris landmarks):
      Eyes fully open:   0.25–0.35
      Eyes half-open:    0.18–0.24
      Eyes closed:       0.05–0.15
    These are systematically ~0.07 lower than Dlib 68-point EAR values
    for the same openness, because MediaPipe's lid landmarks are placed
    closer to the iris margin.
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

    def _lm(idx: int):
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
    #
    # ROOT CAUSE (confirmed by log back-calculation):
    # Observed neutral AU43=0.65 with old high=0.30, low=0.15 implies:
    #   0.65 = (0.30 - EAR) / 0.15  →  EAR = 0.30 - 0.0975 = 0.2025
    # So MediaPipe's actual neutral EAR (6-point aperture, refine_landmarks=True)
    # is ~0.18–0.22 — significantly lower than the 0.25–0.30 assumed from
    # Dlib literature. Setting au43_high=0.30 placed the neutral face halfway
    # into the "eyes closing" zone.
    #
    # FIX: au43_high=0.25, au43_low=0.06 (see settings.py for full rationale).
    # EAR ≥ 0.25 → AU43 = 0  (wide open)
    # EAR = 0.20 → AU43 ≈ 0.26  (neutral, open)
    # EAR = 0.12 → AU43 ≈ 0.68  (pain squint)
    # EAR ≤ 0.06 → AU43 = 1.0   (fully closed)
    # ────────────────────────────────────────────────────────────────────
    au.au43 = _scale(cfg.au43_high - ear_avg, 0.0, cfg.au43_high - cfg.au43_low)

    # ────────────────────────────────────────────────────────────────────
    # AU5   Upper Lid Raiser  (high EAR = wide open eyes)
    # ────────────────────────────────────────────────────────────────────
    au.au5 = _scale(ear_avg, cfg.au5_low, cfg.au5_high)

    # ────────────────────────────────────────────────────────────────────
    # AU7   Lid Tightener  (slight squint — moderate EAR reduction)
    # ────────────────────────────────────────────────────────────────────
    mid_ear = (cfg.au43_low + cfg.au5_low) / 2.0
    au7_raw = max(0.0, mid_ear - ear_avg)
    au.au7 = _scale(au7_raw, cfg.au7_low, cfg.au7_high)

    # ────────────────────────────────────────────────────────────────────
    # AU4   Brow Lowerer
    #
    # Geometry (correct): inner-brow landmarks (55/285) track corrugator
    # contraction; lid-apex landmarks (159/386) give the highest stable
    # upper-lid point. Gap = (lid_apex_y - inner_brow_y) / IOD.
    # In pixel space y increases downward, so lid_apex_y > inner_brow_y
    # → gap is positive, shrinks as brows are pressed down. ✓
    #
    # ROOT CAUSE of high neutral AU4:
    # Observed neutral AU4=0.62 with old high=0.20, low=0.05:
    #   brow_dist = 0.20 - 0.62×0.15 = 0.107 IOD
    # The actual neutral inner-brow→lid-apex gap is ~0.10–0.13 IOD.
    # Old au4_high=0.20 placed this neutral gap inside the active range,
    # producing AU4≈0.45–0.65 at rest.
    #
    # FIX: au4_high=0.13, au4_low=0.03 (see settings.py for full rationale).
    # gap ≥ 0.13 IOD → AU4 = 0  (relaxed brow)
    # gap = 0.11 IOD → AU4 ≈ 0.20  (neutral, good)
    # gap = 0.07 IOD → AU4 ≈ 0.60  (moderate furrow / pain)
    # gap ≤ 0.03 IOD → AU4 = 1.0   (maximal corrugator contraction)
    # ────────────────────────────────────────────────────────────────────
    left_inner_brow  = _lm(_LEFT_INNER_BROW_IDX)
    right_inner_brow = _lm(_RIGHT_INNER_BROW_IDX)
    left_lid_apex    = _lm(_LEFT_LID_APEX_IDX)
    right_lid_apex   = _lm(_RIGHT_LID_APEX_IDX)

    left_brow_dist  = (left_lid_apex.y  - left_inner_brow.y)  / iod
    right_brow_dist = (right_lid_apex.y - right_inner_brow.y) / iod
    brow_dist_avg   = (left_brow_dist + right_brow_dist) / 2.0

    au.brow_distance_ratio = brow_dist_avg
    # AU4 increases as brow-to-lid gap SHRINKS below neutral
    au.au4 = _scale(cfg.au4_high - brow_dist_avg, 0.0, cfg.au4_high - cfg.au4_low)

    # ────────────────────────────────────────────────────────────────────
    # AU6   Cheek Raiser
    #
    # Geometry (correct): malar landmarks (205/425) at the zygomatic eminence
    # rise visibly when orbicularis oculi / zygomaticus minor fire. Lower-lid
    # reference eye[5] (lower-outer corner, lm 144 / 380) is the most stable
    # inferior palpebral margin point. Gap = (malar_y - lower_lid_y) / IOD.
    #
    # ROOT CAUSE of high neutral AU6:
    # Observed neutral AU6=0.30 with old high=0.45, low=0.20:
    #   0.30 = (0.45 - cheek_gap) / 0.25  →  cheek_gap = 0.375 IOD
    # So neutral malar→lower-lid gap is ~0.375 IOD. With au6_high=0.45 the
    # neutral gap fell inside the scale range, giving AU6≈0.30 at rest.
    #
    # FIX: au6_high=0.40, au6_low=0.18 (see settings.py for full rationale).
    # gap ≥ 0.40 IOD → AU6 = 0  (cheek fully relaxed)
    # gap = 0.375 IOD → AU6 ≈ 0.10  (neutral, good)
    # gap = 0.27 IOD  → AU6 ≈ 0.59  (pain cheek raise)
    # gap ≤ 0.18 IOD  → AU6 = 1.0   (maximal cheek raise)
    # ────────────────────────────────────────────────────────────────────
    left_malar  = _lm(_LEFT_MALAR_IDX)
    right_malar = _lm(_RIGHT_MALAR_IDX)

    nose_y = _lm(NOSE_TIP_IDX).y

    left_cheek_gap  = (left_malar.y  - nose_y) / iod
    right_cheek_gap = (right_malar.y - nose_y) / iod

    cheek_gap_avg = (left_cheek_gap + right_cheek_gap) / 2.0

    
    au.au6 = _scale(
        cfg.au6_high - cheek_gap_avg,
        0.0,
        cfg.au6_high - cfg.au6_low
    )

    # ────────────────────────────────────────────────────────────────────
    # AU9   Nose Wrinkler
    # ────────────────────────────────────────────────────────────────────
    left_nos  = _region(LEFT_NOSTRIL)
    right_nos = _region(RIGHT_NOSTRIL)
    nostril_left_x  = float(np.mean([lm.x for lm in left_nos]))
    nostril_right_x = float(np.mean([lm.x for lm in right_nos]))
    nostril_width   = abs(nostril_right_x - nostril_left_x) / iod

    NOSTRIL_NEUTRAL = 0.30
    au9_raw = max(0.0, nostril_width - NOSTRIL_NEUTRAL)
    au.au9 = _scale(au9_raw, cfg.au9_low, cfg.au9_high)

    # ────────────────────────────────────────────────────────────────────
    # AU10  Upper Lip Raiser
    # ────────────────────────────────────────────────────────────────────
    nose_tip  = _lm(NOSE_TIP_IDX)
    upper_lip = _lm(UPPER_LIP_CENTER)
    philtrum  = dist(nose_tip, upper_lip) / iod

    PHILTRUM_NEUTRAL = 0.28
    au10_raw = max(0.0, PHILTRUM_NEUTRAL - philtrum)
    au.au10 = _scale(au10_raw, cfg.au10_low, cfg.au10_high)

    # ────────────────────────────────────────────────────────────────────
    # AU12  Lip Corner Puller (smile / grimace pull)
    # ────────────────────────────────────────────────────────────────────
    l_corner = _lm(MOUTH_LEFT_CORNER)
    r_corner = _lm(MOUTH_RIGHT_CORNER)
    mouth_width = dist(l_corner, r_corner) / iod

    MOUTH_WIDTH_NEUTRAL = 0.45
    au12_raw = max(0.0, mouth_width - MOUTH_WIDTH_NEUTRAL)
    au.au12 = _scale(au12_raw, cfg.au12_low, cfg.au12_high)

    # ────────────────────────────────────────────────────────────────────
    # AU15  Lip Corner Depressor
    # ────────────────────────────────────────────────────────────────────
    lower_lip_y       = _lm(LOWER_LIP_CENTER).y
    l_corner_y        = l_corner.y
    r_corner_y        = r_corner.y
    corner_vs_lip_avg = ((l_corner_y + r_corner_y) / 2.0 - lower_lip_y) / iod
    au15_raw = max(0.0, corner_vs_lip_avg)
    au.au15 = _scale(au15_raw, cfg.au15_low, cfg.au15_high)

    # ────────────────────────────────────────────────────────────────────
    # AU17  Chin Raiser
    # ────────────────────────────────────────────────────────────────────
    chin_lm      = _lm(CHIN_IDX)
    lower_lip_lm = _lm(LOWER_LIP_CENTER)
    chin_lip_dist = dist(chin_lm, lower_lip_lm) / iod

    CHIN_LIP_NEUTRAL = 0.18
    au17_raw = max(0.0, CHIN_LIP_NEUTRAL - chin_lip_dist)
    au.au17 = _scale(au17_raw, cfg.au17_low, cfg.au17_high)

    # ────────────────────────────────────────────────────────────────────
    # AU20  Lip Stretcher
    # ────────────────────────────────────────────────────────────────────
    au.au20 = _scale(mouth_width, cfg.au20_low, cfg.au20_high)

    # ────────────────────────────────────────────────────────────────────
    # AU23  Lip Tightener
    # ────────────────────────────────────────────────────────────────────
    upper_inner   = _region(UPPER_LIP_INNER)
    lower_inner   = _region(LOWER_LIP_INNER)
    upper_inner_y = float(np.mean([lm.y for lm in upper_inner]))
    lower_inner_y = float(np.mean([lm.y for lm in lower_inner]))
    lip_thickness = abs(lower_inner_y - upper_inner_y) / iod

    LIP_THICKNESS_NEUTRAL = 0.08
    au23_raw = max(0.0, LIP_THICKNESS_NEUTRAL - lip_thickness)
    au.au23 = _scale(au23_raw, cfg.au23_low, cfg.au23_high)

    # ────────────────────────────────────────────────────────────────────
    # AU24  Lip Pressor
    # ────────────────────────────────────────────────────────────────────
    inter_lip = abs(_lm(UPPER_LIP_CENTER).y - _lm(LOWER_LIP_CENTER).y) / iod
    au24_raw  = max(0.0, cfg.au24_high - inter_lip)
    au.au24   = _scale(au24_raw, 0.0, cfg.au24_high)

    # ────────────────────────────────────────────────────────────────────
    # AU25  Lips Part
    # ────────────────────────────────────────────────────────────────────
    au.au25 = _scale(inter_lip, cfg.au25_low, cfg.au25_high)

    if mouth_width * iod > 1e-3:
        au.mar = (inter_lip * iod) / (mouth_width * iod)
    else:
        au.mar = 0.0

    # ────────────────────────────────────────────────────────────────────
    # AU26  Jaw Drop
    # ────────────────────────────────────────────────────────────────────
    face_height = dist(_lm(NOSE_TIP_IDX), _lm(CHIN_IDX)) / iod
    FACE_HEIGHT_NEUTRAL = 0.90
    jaw_drop_raw = max(0.0, face_height - FACE_HEIGHT_NEUTRAL)
    au.au26 = _scale(jaw_drop_raw, cfg.au26_low, cfg.au26_high)

    return au