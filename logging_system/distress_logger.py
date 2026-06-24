"""
logging_system/distress_logger.py
===================================
Research-grade logging: CSV row per frame + JSONL episode log.

CSV columns
-----------
timestamp, frame, global_distress, confidence, pain_index, fear_index,
fatigue_index, agitation_index, tension_index, respiratory_index,
eye_closure_index, au4..au43, active_episodes, baseline_ready

JSONL
-----
One JSON object per flush cycle (every N frames) containing:
  - frame metadata
  - full AU snapshot
  - full clinical indices
  - active and completed episodes
"""

from __future__ import annotations
import csv
import json
import os
import time
from pathlib import Path
from typing import List
from backend_sender import send_to_backend

from action_units.au_estimator import AUFrame
from clinical_metrics.indices import ClinicalIndices
from confidence.confidence_estimator import ConfidenceResult
from episodes.episode_detector import Episode
from config.settings import CFG


_CSV_FIELDNAMES = [
    "timestamp","subject_id", "frame",
    "global_distress", "face_confidence",
    "pain_index", "fear_index", "fatigue_index",
    "agitation_index", "tension_index", "respiratory_index",
    # AU intensities
    "au4", "au5", "au6", "au7", "au9", "au10",
    "au12", "au15", "au17", "au20", "au23", "au24", "au25", "au26", "au43",
    # Derived
    "mar",
    "episode_pain","episode_agitation","episode_fatigue","episode_fear","episode_eye_closure",
    # Gaze Features
    "gaze_yaw","gaze_pitch", "gaze_variance", "fixation_duration", "recent_saccades","mean_gaze_speed","max_gaze_speed", "eye_contact_ratio",
    "behavior_state",
    "baseline_ready",
]


class DistressLogger:
    """
    Writes per-frame CSV and per-batch JSONL logs.
    Thread-safe for single-threaded use (no locks needed for capture loop).
    """

    def __init__(self,subject_id="subject_01") -> None:
        cfg = CFG.logging
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        self._csv_path  = out_dir / cfg.csv_filename
        self._json_path = out_dir / cfg.json_filename
        self._flush_interval = cfg.flush_interval_frames
        self._frame_count = 0
        self.subject_id = subject_id

        # Open CSV
        self._csv_file = open(self._csv_path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(
            self._csv_file, fieldnames=_CSV_FIELDNAMES
        )
        self._csv_writer.writeheader()

        # Open JSONL (append mode)
        self._json_file = open(self._json_path, "a", encoding="utf-8")

        print(f"[Logger] CSV  → {self._csv_path}")
        print(f"[Logger] JSONL → {self._json_path}")
    

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def log_frame(
        self,
        frame_idx: int,
        au: AUFrame,
        ci: ClinicalIndices,
        confidence: ConfidenceResult,
        active_episodes: List[Episode],
        completed_episodes: List[Episode],
        baseline_ready: bool,

        gaze_yaw: float = 0.0,
        gaze_pitch: float = 0.0,
        gaze_data: dict | None = None,
        behavior_state: str = "alert",
    ) -> None:
        """Log a single frame to CSV (and optionally JSONL)."""
        ts = time.time()
        self._frame_count += 1

        # ── CSV row ──────────────────────────────────────────────────────

        active_types = {
            ep.episode_type.name
            for ep in active_episodes
        }
                
        row = {
            "timestamp":          round(ts, 4),
            "subject_id": self.subject_id,
            "frame":              frame_idx,
            "global_distress":    round(ci.global_distress, 2),
            "face_confidence":         round(confidence.overall, 3),
            "pain_index":         round(ci.pain_index, 2),
            "fear_index":         round(ci.fear_index, 2),
            "fatigue_index":      round(ci.fatigue_index, 2),
            "agitation_index":    round(ci.agitation_index, 2),
            "tension_index":      round(ci.tension_index, 2),
            "respiratory_index":  round(ci.respiratory_index, 2),

            "gaze_yaw": round(gaze_yaw, 4),
            "gaze_pitch": round(gaze_pitch, 4),

            "gaze_variance": (
                round(gaze_data.get("gaze_variance", 0), 4)
                if gaze_data else 0
            ),

            "fixation_duration": (
                round(gaze_data.get("fixation_duration", 0), 4)
                if gaze_data else 0
            ),

            "recent_saccades": (
                gaze_data.get("recent_saccades", 0)
                if gaze_data else 0
            ),

            "mean_gaze_speed": (
                round(gaze_data.get("mean_gaze_speed", 0), 4)
                if gaze_data else 0
            ),

            "max_gaze_speed": (
                round(gaze_data.get("max_gaze_speed", 0), 4)
                if gaze_data else 0
            ),

            "eye_contact_ratio": (
                round(gaze_data.get("eye_contact_ratio", 0), 4)
                if gaze_data else 0
            ),

            "behavior_state": behavior_state,
            # AU intensities
            "au4":  round(au.au4,  4),
            "au5":  round(au.au5,  4),
            "au6":  round(au.au6,  4),
            "au7":  round(au.au7,  4),
            "au9":  round(au.au9,  4),
            "au10": round(au.au10, 4),
            "au12": round(au.au12, 4),
            "au15": round(au.au15, 4),
            "au17": round(au.au17, 4),
            "au20": round(au.au20, 4),
            "au23": round(au.au23, 4),
            "au24": round(au.au24, 4),
            "au25": round(au.au25, 4),
            "au26": round(au.au26, 4),
            "au43": round(au.au43, 4),
            "mar": round(au.mar, 4),

            "episode_pain": int("PAIN" in active_types),

            "episode_agitation": int("AGITATION" in active_types),

            "episode_fatigue": int("FATIGUE" in active_types),

            "episode_fear": int("FEAR" in active_types),

            "episode_eye_closure": int("EYE_CLOSURE" in active_types),

            "baseline_ready": int(baseline_ready),
        }
        backend_payload = {
            "timestamp_unix": ts,

            "global_distress": row["global_distress"],

            "pain_index": row["pain_index"],
            "fear_index": row["fear_index"],
            "fatigue_index": row["fatigue_index"],
            "agitation_index": row["agitation_index"],
            "tension_index": row["tension_index"],
            "respiratory_index": row["respiratory_index"],

            "behavior_state": row["behavior_state"],

            "gaze_yaw": row["gaze_yaw"],
            "gaze_pitch": row["gaze_pitch"],

            "gaze_variance": row["gaze_variance"],
            "fixation_duration": row["fixation_duration"],

            "recent_saccades": row["recent_saccades"],

            "eye_contact_ratio": row["eye_contact_ratio"],

            "episode_pain": row["episode_pain"],
            "episode_agitation": row["episode_agitation"],
            "episode_fatigue": row["episode_fatigue"],
            "episode_fear": row["episode_fear"],
            "episode_eye_closure": row["episode_eye_closure"],

            "face_confidence": row["face_confidence"]
        }

        send_to_backend(
            backend_payload
        )
        self._csv_writer.writerow(row)

        # ── JSONL flush every N frames ────────────────────────────────────
        if self._frame_count % self._flush_interval == 0:
            record = {
                "timestamp":  ts,
                "frame":      frame_idx,
                "au":         au.as_dict(),
                "clinical":   ci.as_dict(),
                "confidence": confidence.as_dict(),
                "active_episodes":    [ep.as_dict() for ep in active_episodes],
                "completed_episodes": [ep.as_dict() for ep in completed_episodes[-5:]],
                "baseline_ready":     baseline_ready,
            }
            self._json_file.write(json.dumps(record) + "\n")
            self._csv_file.flush()
            self._json_file.flush()
    

    def close(self) -> None:
        """Flush and close all file handles."""
        self._csv_file.flush()
        self._json_file.flush()
        self._csv_file.close()
        self._json_file.close()
        print(f"[Logger] Session saved → {self._csv_path.parent}")
