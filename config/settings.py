"""
config/settings.py
==================
Central configuration for the Clinical Facial Distress Detection System.
All thresholds, weights, and constants live here — no magic numbers in code.
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
    max_num_faces: int = 4           # detect up to 4; only track patient
    refine_landmarks: bool = True
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


# ──────────────────────────────────────────────
# Action Unit geometric thresholds
# Each AU has a (min_raw, max_raw) range that gets
# linearly mapped to [0.0, 1.0].
# ──────────────────────────────────────────────
@dataclass(frozen=True)
class AUConfig:
    # AU4  Brow Lowerer: brow-to-eye ratio; lower = more raised/pressed
    au4_low: float = 0.20
    au4_high: float = 0.50

    # AU5  Upper Lid Raiser: EAR threshold; higher EAR = lid raised
    au5_low: float = 0.20
    au5_high: float = 0.40

    # AU6  Cheek Raiser: cheek-to-eye vertical ratio
    au6_low: float = 0.05
    au6_high: float = 0.25

    # AU7  Lid Tightener: inter-lid distance ratio (inverse of EAR)
    au7_low: float = 0.10
    au7_high: float = 0.30

    # AU9  Nose Wrinkler: nostril width ratio deviation
    au9_low: float = 0.00
    au9_high: float = 0.15

    # AU10 Upper Lip Raiser: upper-lip-to-nose distance ratio
    au10_low: float = 0.10
    au10_high: float = 0.35

    # AU12 Lip Corner Puller (smile): corner displacement up
    au12_low: float = 0.00
    au12_high: float = 0.20

    # AU15 Lip Corner Depressor: corner displacement down
    au15_low: float = 0.00
    au15_high: float = 0.20

    # AU17 Chin Raiser: chin-to-lip ratio
    au17_low: float = 0.00
    au17_high: float = 0.15

    # AU20 Lip Stretcher: lip width ratio
    au20_low: float = 0.30
    au20_high: float = 0.60

    # AU23 Lip Tightener: lip thickness inverse
    au23_low: float = 0.00
    au23_high: float = 0.20

    # AU24 Lip Pressor: inter-lip gap (very small = pressed)
    au24_low: float = 0.00
    au24_high: float = 0.05

    # AU25 Lips Part: inter-lip distance
    au25_low: float = 0.02
    au25_high: float = 0.25

    # AU26 Jaw Drop: jaw-to-chin distance ratio
    au26_low: float = 0.05
    au26_high: float = 0.35

    # AU43 Eye Closure: EAR inverted (closed = high)
    au43_low: float = 0.10
    au43_high: float = 0.32


# ──────────────────────────────────────────────
# PSPI-inspired Pain Index
# Pain = AU4 + max(AU6, AU7) + max(AU9, AU10) + AU43
# ──────────────────────────────────────────────
@dataclass(frozen=True)
class PainIndexConfig:
    au4_weight: float = 1.0
    au6_au7_weight: float = 1.0
    au9_au10_weight: float = 1.0
    au43_weight: float = 1.0
    max_raw: float = 4.0            # sum of 4 components each [0,1]


# ──────────────────────────────────────────────
# Composite Clinical Indices (all → [0,1])
# ──────────────────────────────────────────────
@dataclass(frozen=True)
class ClinicalIndexConfig:
    # Fear/Anxiety: AU5 + AU20 + AU26 + asymmetry
    fear_weights: Dict[str, float] = field(
        default_factory=lambda: {"au5": 0.35, "au20": 0.30, "au26": 0.20, "asymmetry": 0.15}
    )

    # Fatigue: AU43 + PERCLOS + landmark_variance_low
    fatigue_weights: Dict[str, float] = field(
        default_factory=lambda: {"au43": 0.45, "perclos": 0.40, "low_variance": 0.15}
    )

    # Agitation: motion_energy + au26 + head_pose_deviation
    agitation_weights: Dict[str, float] = field(
        default_factory=lambda: {"motion": 0.40, "au26": 0.30, "head_pose": 0.30}
    )

    # Facial Tension: AU4 + AU7 + AU23 + AU24
    tension_weights: Dict[str, float] = field(
        default_factory=lambda: {"au4": 0.35, "au7": 0.25, "au23": 0.25, "au24": 0.15}
    )

    # Respiratory Distress Proxy: AU26 rate + nostril flare (AU9) + jaw rhythm
    respiratory_weights: Dict[str, float] = field(
        default_factory=lambda: {"au26_variance": 0.50, "au9": 0.30, "jaw_rhythm": 0.20}
    )

    # Global Distress: weighted combination of sub-indices
    global_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "pain_index": 0.35,
            "fear_index": 0.15,
            "fatigue_index": 0.15,
            "agitation_index": 0.15,
            "tension_index": 0.10,
            "respiratory_index": 0.10,
        }
    )


# ──────────────────────────────────────────────
# Baseline Collection
# ──────────────────────────────────────────────
@dataclass(frozen=True)
class BaselineConfig:
    collection_seconds: float = 30.0   # 30-second resting baseline
    min_samples: int = 150             # minimum frames needed
    mad_scale: float = 1.4826          # consistency factor for MAD → σ


# ──────────────────────────────────────────────
# Temporal Buffers (at 30 fps)
# ──────────────────────────────────────────────
@dataclass(frozen=True)
class TemporalConfig:
    fps: int = 30
    buffer_10s: int = 300
    buffer_30s: int = 900
    buffer_60s: int = 1800
    ema_alpha: float = 0.15            # smoothing factor for EMA


# ──────────────────────────────────────────────
# Episode Detection
# ──────────────────────────────────────────────
@dataclass(frozen=True)
class EpisodeConfig:
    pain_threshold: float = 70.0       # global score to trigger pain episode
    agitation_threshold: float = 60.0
    fatigue_threshold: float = 70.0
    eye_closure_threshold: float = 0.70  # PERCLOS
    min_episode_frames: int = 45       # avoid single-frame spikes
    cooldown_frames: int = 60          # frames before new episode can start


# ──────────────────────────────────────────────
# Confidence Estimation
# ──────────────────────────────────────────────
@dataclass(frozen=True)
class ConfidenceConfig:
    min_landmark_quality: float = 0.40
    max_yaw_deg: float = 35.0
    max_pitch_deg: float = 30.0
    max_roll_deg: float = 25.0
    occlusion_penalty: float = 0.30
    tracking_stability_window: int = 15   # frames


# ──────────────────────────────────────────────
# Multi-Face Patient Tracking
# ──────────────────────────────────────────────
@dataclass(frozen=True)
class TrackingConfig:
    iou_threshold: float = 0.40
    max_lost_frames: int = 30          # frames before patient ID is reset
    patient_lock_frames: int = 60      # frames to confirm patient identity


# ──────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────
@dataclass(frozen=True)
class VisualizationConfig:
    panel_width: int = 320
    panel_alpha: float = 0.75
    font_scale: float = 0.48
    font_thickness: int = 1
    bar_height: int = 10
    bar_width: int = 180
    line_height: int = 19

    # Colour palette (BGR)
    color_ok: tuple = (80, 220, 80)
    color_mild: tuple = (60, 200, 255)
    color_moderate: tuple = (40, 140, 255)
    color_severe: tuple = (40, 40, 255)
    color_text: tuple = (230, 230, 230)
    color_label: tuple = (170, 200, 255)
    color_panel_bg: tuple = (20, 20, 30)
    color_episode: tuple = (0, 80, 220)


# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
@dataclass(frozen=True)
class LoggingConfig:
    output_dir: str = "outputs"
    csv_filename: str = "distress_log.csv"
    json_filename: str = "distress_log.jsonl"
    flush_interval_frames: int = 30    # write to disk every N frames


# ──────────────────────────────────────────────
# Master Config Container
# ──────────────────────────────────────────────
@dataclass(frozen=True)
class SystemConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    face_mesh: FaceMeshConfig = field(default_factory=FaceMeshConfig)
    au: AUConfig = field(default_factory=AUConfig)
    pain_index: PainIndexConfig = field(default_factory=PainIndexConfig)
    clinical: ClinicalIndexConfig = field(default_factory=ClinicalIndexConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


# Singleton instance used throughout the project
CFG = SystemConfig()
