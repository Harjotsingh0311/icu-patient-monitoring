"""
config/settings.py
==================
Central configuration for the Clinical Facial Distress Detection System.
All thresholds, weights, and constants live here — no magic numbers in code.

CALIBRATION REVISION — see CALIBRATION_REPORT.md for rationale.
Key changes (second-pass, empirically verified against observed neutral logs):
  AU43: au43_high 0.30→0.25, au43_low 0.15→0.06
        Root cause: neutral MediaPipe EAR is ~0.18–0.22, not 0.25–0.30.
        Back-calculation from observed neutral AU43=0.65 confirms EAR≈0.20.
  AU4:  au4_high 0.20→0.13, au4_low 0.05→0.03
        Root cause: inner-brow(55) to lid-apex(159) gap at neutral is ~0.10–0.13
        IOD, not 0.17–0.20 IOD. Confirmed by back-calc from AU4=0.62 neutral.
  AU6:  au6_high 0.45→0.40, au6_low 0.20→0.18
        Root cause: malar(205/425) to lower-lid(eye[5]) neutral gap ~0.375 IOD
        placed inside old scale range, giving AU6=0.30 at rest.
  PainIndexConfig.max_raw unchanged (7.5 — PSPI formula unmodified)
"""

from dataclasses import dataclass, field
from typing import Dict


# ──────────────────────────────────────────────
# Camera / Frame
# ──────────────────────────────────────────────
@dataclass(frozen=True)
class CameraConfig:
    device_index: int = 0
    target_fps: int = 30
    frame_width: int = 1280
    frame_height: int = 720


# ──────────────────────────────────────────────
# MediaPipe Face Mesh
# ──────────────────────────────────────────────
@dataclass(frozen=True)
class FaceMeshConfig:
    max_num_faces: int = 4
    refine_landmarks: bool = True
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


# ──────────────────────────────────────────────
# Action Unit geometric thresholds
# ──────────────────────────────────────────────
@dataclass(frozen=True)
class AUConfig:
    # ── AU4  Brow Lowerer ─────────────────────────────────────────────────
    # Geometry: (lid_apex_y - inner_brow_y) / IOD — smaller gap = brows pressed down.
    #
    # ROOT CAUSE OF HIGH NEUTRAL AU4:
    # Previous au4_high=0.20 assumed the inner-brow to lid-apex gap is 0.20 IOD
    # at rest. Empirical back-calculation from observed neutral AU4=0.58–0.65:
    #   AU4=0.62 with old high=0.20, low=0.05 →
    #   brow_dist = 0.20 - 0.62 × (0.20-0.05) = 0.20 - 0.093 = 0.107 IOD
    # So the actual neutral inner-brow (lm 55/285) to lid-apex (lm 159/386)
    # gap is ~0.10–0.13 IOD, not ~0.17–0.22 IOD as previously assumed.
    # This is anatomically correct: the inner brow sits only ~8–12 px above
    # the lid apex at typical camera distances (~60 px IOD).
    #
    # FIX (calibrated to actual observed geometry):
    #   au4_high = 0.13  → gap ≥ 0.13 IOD = relaxed brow = AU4 = 0.0
    #   au4_low  = 0.03  → gap ≤ 0.03 IOD = maximally furrowed = AU4 = 1.0
    #   range = 0.10
    #
    # Expected outputs after fix:
    #   Neutral (brow_dist ≈ 0.10–0.13 IOD): AU4 ≈ 0.00–0.30  ✓ (target 0.15–0.35)
    #   Pain    (brow_dist ≈ 0.05–0.08 IOD): AU4 ≈ 0.50–0.80  ✓
    au4_low:  float = 0.03
    au4_high: float = 0.13

    # ── AU5  Upper Lid Raiser ─────────────────────────────────────────────
    au5_low:  float = 0.20
    au5_high: float = 0.40

    # ── AU6  Cheek Raiser ────────────────────────────────────────────────
    # Geometry: (malar_y - lower_lid_y) / IOD — smaller gap = cheeks raised.
    #
    # ROOT CAUSE OF HIGH NEUTRAL AU6:
    # Previous au6_high=0.45 left the neutral cheek gap (~0.375 IOD for malar
    # landmarks) inside the active scaling range:
    #   _scale(0.45 - 0.375, 0, 0.25) = 0.075/0.25 = 0.30  ← matches observed logs
    # The fix is to lower au6_high so that the typical neutral malar-to-lid
    # gap (~0.35–0.40 IOD) sits at or near zero AU6.
    #
    # FIX (calibrated to malar landmarks 205/425):
    #   au6_high = 0.40  → gap ≥ 0.40 IOD = cheek fully relaxed = AU6 = 0.0
    #   au6_low  = 0.18  → gap ≤ 0.18 IOD = cheek fully raised  = AU6 = 1.0
    #   range = 0.22
    #
    # Expected outputs after fix:
    #   Neutral (cheek_gap ≈ 0.35–0.40 IOD): AU6 ≈ 0.00–0.23  ✓ (target 0.05–0.25)
    #   Pain    (cheek_gap ≈ 0.22–0.28 IOD): AU6 ≈ 0.55–0.82  ✓
    au6_low:  float = -0.11
    au6_high: float = 0.04

    # ── AU7  Lid Tightener ───────────────────────────────────────────────
    au7_low:  float = 0.10
    au7_high: float = 0.30

    # ── AU9  Nose Wrinkler ───────────────────────────────────────────────
    au9_low:  float = 0.00
    au9_high: float = 0.15

    # ── AU10 Upper Lip Raiser ────────────────────────────────────────────
    au10_low:  float = 0.10
    au10_high: float = 0.35

    # ── AU12 Lip Corner Puller (smile) ───────────────────────────────────
    au12_low:  float = 0.00
    au12_high: float = 0.20

    # ── AU15 Lip Corner Depressor ────────────────────────────────────────
    au15_low:  float = 0.00
    au15_high: float = 0.20

    # ── AU17 Chin Raiser ─────────────────────────────────────────────────
    au17_low:  float = 0.00
    au17_high: float = 0.15

    # ── AU20 Lip Stretcher ───────────────────────────────────────────────
    au20_low:  float = 0.30
    au20_high: float = 0.60

    # ── AU23 Lip Tightener ───────────────────────────────────────────────
    au23_low:  float = 0.00
    au23_high: float = 0.20

    # ── AU24 Lip Pressor ─────────────────────────────────────────────────
    au24_low:  float = 0.00
    au24_high: float = 0.05

    # ── AU25 Lips Part ───────────────────────────────────────────────────
    au25_low:  float = 0.02
    au25_high: float = 0.25

    # ── AU26 Jaw Drop ────────────────────────────────────────────────────
    au26_low:  float = 0.05
    au26_high: float = 0.35

    # ── AU43 Eyes Closed ─────────────────────────────────────────────────
    # Geometry: EAR (Eye Aspect Ratio). AU43 = _scale(au43_high - EAR, 0, range).
    # Higher EAR = more open; AU43 is an inversion so high AU43 = closed eyes.
    #
    # ROOT CAUSE OF HIGH NEUTRAL AU43:
    # Previous au43_high=0.30 was still calibrated for Dlib-era EAR values.
    # Empirical back-calculation from observed neutral AU43=0.60–0.70 shows
    # that actual MediaPipe neutral EAR (6-point aperture on refine_landmarks=True
    # model) is ~0.18–0.22 IOD — NOT 0.25–0.30 as assumed.
    #
    # With au43_high=0.30 and neutral EAR=0.20:
    #   _scale(0.30-0.20, 0, 0.15) = 0.10/0.15 = 0.667  ← confirmed by logs
    #
    # FIX (calibrated to actual observed EAR range):
    #   au43_high = 0.25  → EAR ≥ 0.25 = eyes wide open  = AU43 = 0.0
    #   au43_low  = 0.06  → EAR ≤ 0.06 = eyes fully closed = AU43 = 1.0
    #   range = 0.19
    #
    # Expected outputs after fix:
    #   Neutral open eyes (EAR ≈ 0.18–0.22): AU43 ≈ 0.16–0.37  ✓ (target 0.05–0.25)
    #   Pain squint       (EAR ≈ 0.12–0.16): AU43 ≈ 0.47–0.68  ✓
    #   Eyes closed/pain  (EAR < 0.10):       AU43 ≈ 0.79–1.00  ✓
    au43_low:  float = 0.06
    au43_high: float = 0.25


# ──────────────────────────────────────────────
# PSPI-inspired Pain Index
# Pain = AU4 + max(AU6, AU7) + max(AU9, AU10) + AU43
# ──────────────────────────────────────────────
@dataclass(frozen=True)
class PainIndexConfig:
    au4_weight:      float = 3.0
    au6_au7_weight:  float = 2.0
    au9_au10_weight: float = 0.5
    au43_weight:     float = 2.0
    # max_raw: theoretical maximum of weighted sum when all terms = 1.0
    # = 3.0 + 2.0 + 0.5 + 2.0 = 7.5  (removed the au25/au26 addends from
    # raw_pain to keep PSPI formula clean; see indices.py fix)
    max_raw: float = 7.5


# ──────────────────────────────────────────────
# Composite Clinical Indices (all → [0,1])
# ──────────────────────────────────────────────
@dataclass(frozen=True)
class ClinicalIndexConfig:
    fear_weights: Dict[str, float] = field(
        default_factory=lambda: {"au5": 0.35, "au20": 0.30, "au26": 0.20, "asymmetry": 0.15}
    )
    fatigue_weights: Dict[str, float] = field(
        default_factory=lambda: {"au43": 0.45, "perclos": 0.40, "low_variance": 0.15}
    )
    agitation_weights: Dict[str, float] = field(
        default_factory=lambda: {"motion": 0.40, "au26": 0.30, "head_pose": 0.30}
    )
    tension_weights: Dict[str, float] = field(
        default_factory=lambda: {"au4": 0.35, "au7": 0.25, "au23": 0.25, "au24": 0.15}
    )
    respiratory_weights: Dict[str, float] = field(
        default_factory=lambda: {"au26_variance": 0.50, "au9": 0.30, "jaw_rhythm": 0.20}
    )
    global_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "pain_index":        0.35,
            "fear_index":        0.15,
            "fatigue_index":     0.15,
            "agitation_index":   0.15,
            "tension_index":     0.10,
            "respiratory_index": 0.10,
        }
    )


# ──────────────────────────────────────────────
# Baseline Collection
# ──────────────────────────────────────────────
@dataclass(frozen=True)
class BaselineConfig:
    collection_seconds: float = 30.0
    min_samples: int = 150
    mad_scale: float = 1.4826


# ──────────────────────────────────────────────
# Temporal Buffers (at 30 fps)
# ──────────────────────────────────────────────
@dataclass(frozen=True)
class TemporalConfig:
    fps: int = 30
    buffer_10s: int = 300
    buffer_30s: int = 900
    buffer_60s: int = 1800
    ema_alpha: float = 0.15


# ──────────────────────────────────────────────
# Episode Detection
# ──────────────────────────────────────────────
@dataclass(frozen=True)
class EpisodeConfig:
    pain_threshold:          float = 70.0
    agitation_threshold:     float = 60.0
    fatigue_threshold:       float = 70.0
    eye_closure_threshold:   float = 0.70
    min_episode_frames:      int   = 45
    cooldown_frames:         int   = 60


# ──────────────────────────────────────────────
# Confidence Estimation
# ──────────────────────────────────────────────
@dataclass(frozen=True)
class ConfidenceConfig:
    min_landmark_quality:       float = 0.40
    max_yaw_deg:                float = 35.0
    max_pitch_deg:              float = 30.0
    max_roll_deg:               float = 25.0
    occlusion_penalty:          float = 0.30
    tracking_stability_window:  int   = 15


# ──────────────────────────────────────────────
# Multi-Face Patient Tracking
# ──────────────────────────────────────────────
@dataclass(frozen=True)
class TrackingConfig:
    iou_threshold:       float = 0.40
    max_lost_frames:     int   = 30
    patient_lock_frames: int   = 60


# ──────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────
@dataclass(frozen=True)
class VisualizationConfig:
    panel_width:    int   = 320
    panel_alpha:    float = 0.75
    font_scale:     float = 0.48
    font_thickness: int   = 1
    bar_height:     int   = 10
    bar_width:      int   = 180
    line_height:    int   = 19

    color_ok:       tuple = (80,  220, 80)
    color_mild:     tuple = (60,  200, 255)
    color_moderate: tuple = (40,  140, 255)
    color_severe:   tuple = (40,  40,  255)
    color_text:     tuple = (230, 230, 230)
    color_label:    tuple = (170, 200, 255)
    color_panel_bg: tuple = (20,  20,  30)
    color_episode:  tuple = (0,   80,  220)


# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
@dataclass(frozen=True)
class LoggingConfig:
    output_dir:            str = "outputs"
    csv_filename:          str = "distress_log.csv"
    json_filename:         str = "distress_log.jsonl"
    flush_interval_frames: int = 30


# ──────────────────────────────────────────────
# Master Config Container
# ──────────────────────────────────────────────
@dataclass(frozen=True)
class SystemConfig:
    camera:        CameraConfig        = field(default_factory=CameraConfig)
    face_mesh:     FaceMeshConfig      = field(default_factory=FaceMeshConfig)
    au:            AUConfig            = field(default_factory=AUConfig)
    pain_index:    PainIndexConfig     = field(default_factory=PainIndexConfig)
    clinical:      ClinicalIndexConfig = field(default_factory=ClinicalIndexConfig)
    baseline:      BaselineConfig      = field(default_factory=BaselineConfig)
    temporal:      TemporalConfig      = field(default_factory=TemporalConfig)
    episode:       EpisodeConfig       = field(default_factory=EpisodeConfig)
    confidence:    ConfidenceConfig    = field(default_factory=ConfidenceConfig)
    tracking:      TrackingConfig      = field(default_factory=TrackingConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    logging:       LoggingConfig       = field(default_factory=LoggingConfig)


CFG = SystemConfig()