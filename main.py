"""
main.py
========
Clinical Facial Distress Detection System — Main Entry Point

Pipeline per frame
------------------
Video Frame
  → Face Detection (MediaPipe FaceMesh, up to 4 faces)
  → Patient Tracking (IoU-based identity persistence)
  → Face Alignment / Landmark Extraction (468 points)
  → Action Unit Estimation (15 AUs, geometric)
  → Patient Baseline Collection (30 s, median/MAD)
  → Temporal Feature Buffering (10 / 30 / 60 s windows)
  → Clinical Index Computation (PSPI pain, fear, fatigue, agitation, tension, respiratory)
  → Confidence Estimation
  → Distress Episode Detection
  → Visualization Overlay
  → CSV + JSONL Logging
"""

from __future__ import annotations
import sys
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

# ── Project modules ──────────────────────────────────────────────────────
from config.settings import CFG

from landmarks.landmark_extractor import (
    extract_landmarks, interocular_distance, LandmarkArray
)
from landmarks.landmark_groups import (
    LEFT_EYE, RIGHT_EYE, LEFT_BROW, RIGHT_BROW,
    JAW_LOWER, LEFT_EYE_OUTER_IDX, RIGHT_EYE_OUTER_IDX,
)

from action_units.au_estimator import estimate_aus, AUFrame

from clinical_metrics.indices import compute_clinical_indices, ClinicalIndices

from baseline.baseline_manager import BaselineManager

from temporal.feature_buffer import TemporalFeatureBuffer

from episodes.episode_detector import EpisodeDetector

from tracking.patient_tracker import PatientTracker, landmarks_to_box

from confidence.confidence_estimator import ConfidenceEstimator, ConfidenceResult

from core.head_pose import estimate_head_pose, head_pose_deviation
from core.motion import compute_motion_energy, compute_facial_asymmetry, compute_perclos

from visualization.overlay import DistressOverlay, draw_landmark_dots
from logging_system.distress_logger import DistressLogger
from eye.gaze_estimator import GazeEstimator
from eye.gaze_tracker import GazeTracker


# ──────────────────────────────────────────────────────────────────────────
# Constants / Sentinel defaults
# ──────────────────────────────────────────────────────────────────────────
_EAR_CLOSURE_THRESHOLD = 0.18    # EAR below this = eye considered closed


def main() -> None:
    cfg = CFG

    # ── MediaPipe ────────────────────────────────────────────────────────
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=cfg.face_mesh.max_num_faces,
        refine_landmarks=cfg.face_mesh.refine_landmarks,
        min_detection_confidence=cfg.face_mesh.min_detection_confidence,
        min_tracking_confidence=cfg.face_mesh.min_tracking_confidence,
    )

    # ── Camera ───────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cfg.camera.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.camera.frame_height)
    cap.set(cv2.CAP_PROP_FPS,          cfg.camera.target_fps)

    if not cap.isOpened():
        print("[ERROR] Cannot open camera. Exiting.")
        sys.exit(1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera {actual_w}×{actual_h} @ {cfg.camera.target_fps} fps")

    # ── Pipeline components ───────────────────────────────────────────────
    tracker     = PatientTracker(actual_w, actual_h)
    baseline    = BaselineManager()
    temp_buf    = TemporalFeatureBuffer()
    ep_detector = EpisodeDetector()
    conf_est    = ConfidenceEstimator()
    overlay     = DistressOverlay()
    logger      = DistressLogger()
    gaze_estimator = GazeEstimator()
    gaze_tracker = GazeTracker()

    # ── State variables ───────────────────────────────────────────────────
    frame_idx: int = 0
    previous_landmarks: LandmarkArray | None = None
    eye_closed_history: deque = deque(maxlen=cfg.temporal.buffer_30s)

    # Default/fallback outputs (shown before first face detection)
    au          = AUFrame()
    ci          = ClinicalIndices()
    confidence  = ConfidenceResult()
    yaw = pitch = roll = 0.0
    gaze_yaw = 0.0
    gaze_pitch = 0.0
    gaze_data = None
    closed_frames = 0

    print("[INFO] Starting distress monitoring. Press ESC to quit.")
    print(f"[INFO] Baseline collection: {cfg.baseline.collection_seconds}s")

    # ── Main loop ─────────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame capture failed — skipping.")
            continue
        behavior_state = "unknown"
        gaze_yaw = 0.0
        gaze_pitch = 0.0
        gaze_data = {}

        frame_idx += 1
        h, w = frame.shape[:2]
        frame_area = max(w * h, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        patient_landmarks: LandmarkArray | None = None

        if results.multi_face_landmarks:
            # ── Patient tracking ─────────────────────────────────────────
            boxes = [
                landmarks_to_box(fl, w, h, i)
                for i, fl in enumerate(results.multi_face_landmarks)
            ]
            patient_box = tracker.update(boxes)

            if patient_box is not None:
                raw_fl = results.multi_face_landmarks[patient_box.face_index]
                patient_landmarks = extract_landmarks(raw_fl, w, h)
                # -------------------------------------------------
            # ---------------------------------------
            # Face Crop for L2CS
            # ---------------------------------------

                xs = [lm.x for lm in patient_landmarks]
                ys = [lm.y for lm in patient_landmarks]

                padding = 20

                x_min = max(0, int(min(xs)) - padding)
                y_min = max(0, int(min(ys)) - padding)

                x_max = min(w, int(max(xs)) + padding)
                y_max = min(h, int(max(ys)) + padding)

                face_crop = frame[
                    y_min:y_max,
                    x_min:x_max
                ]

                if face_crop.size > 0:

                    try:

                        gaze_yaw, gaze_pitch = (
                            gaze_estimator.estimate(face_crop)
                        )

                        gaze_data = gaze_tracker.update(
                            gaze_yaw,
                            gaze_pitch,
                            eye_closed
                            
                        )
                        
                        

                    except Exception as e:

                        print("L2CS Error:", e)

                # ── IOD ─────────────────────────────────────────────────
                iod = interocular_distance(patient_landmarks)

                # ── Action Units ─────────────────────────────────────────
                au = estimate_aus(patient_landmarks, iod=iod)

                # ── Head pose ─────────────────────────────────────────────
                yaw, pitch, roll = estimate_head_pose(patient_landmarks, w, h)
                pose_dev = head_pose_deviation(yaw, pitch, roll)

                # ── Motion energy ─────────────────────────────────────────
                motion = compute_motion_energy(patient_landmarks, previous_landmarks)
                previous_landmarks = patient_landmarks

                # ── Asymmetry ─────────────────────────────────────────────
                asymmetry = compute_facial_asymmetry(patient_landmarks)

                # ── Eye closure & PERCLOS ─────────────────────────────────
                eye_closed = au.ear_avg < _EAR_CLOSURE_THRESHOLD
                scanning_cooldown = 0
                if eye_closed:

                    closed_frames += 1

                    scanning_cooldown = 60

                else:

                    closed_frames = 0

                    if scanning_cooldown > 0:
                        scanning_cooldown -= 1
                eye_closed_history.append(int(eye_closed))
                perclos = compute_perclos(eye_closed_history)


                behavior_state = "alert"

                

                if closed_frames > 30:
                    behavior_state = "eye_closed"

                elif perclos > 40:
                    behavior_state = "drowsy"

                elif gaze_data and gaze_data["fixation_duration"] > 2:
                    behavior_state = "fixation"

                elif (
                    gaze_data
                    and closed_frames == 0
                    and scanning_cooldown == 0
                    and gaze_data["recent_saccades"] > 18
                ):
                    behavior_state = "scanning"

                # ── Temporal buffer push ──────────────────────────────────
                au_dict = au.as_dict()
                au_dict["perclos"]      = perclos
                au_dict["motion_energy"] = motion
                temp_buf.push(au_dict)

                au26_var   = temp_buf.au26_variance(window_seconds=5.0)
                jaw_rhythm = temp_buf.jaw_rhythm_score()

                # ── Baseline update ───────────────────────────────────────
                if not baseline.ready:
                    baseline.update(au_dict)

                # ── Clinical indices ──────────────────────────────────────
                ci = compute_clinical_indices(
                    au=au,
                    perclos=perclos,
                    motion_energy=motion,
                    head_pose_deviation=pose_dev,
                    asymmetry=asymmetry,
                    au26_variance=au26_var,
                    jaw_rhythm=jaw_rhythm,
                    baseline=baseline.baseline if baseline.ready else None,
                )

                # Push clinical indices into temporal buffer
                temp_buf.push(ci.as_dict())

                # ── Confidence ────────────────────────────────────────────
                face_area_frac = patient_box.area / frame_area
                confidence = conf_est.update(
                    landmarks=patient_landmarks,
                    yaw_deg=yaw,
                    pitch_deg=pitch,
                    roll_deg=roll,
                    patient_confirmed=tracker.patient_confirmed,
                    face_area_fraction=face_area_frac,
                )

                # ── Episode detection ─────────────────────────────────────
                ep_detector.update(
                    frame_idx=frame_idx,
                    pain_score=ci.pain_index,
                    agitation_score=ci.agitation_index,
                    eye_closure_score=ci.eye_closure_index,
                    fatigue_score=ci.fatigue_index,
                    fear_score=ci.fear_index,
                )

                # ── Logging ───────────────────────────────────────────────
                logger.log_frame(
                    frame_idx=frame_idx,
                    au=au,
                    ci=ci,
                    confidence=confidence,
                    active_episodes=ep_detector.active_episodes,
                    completed_episodes=ep_detector.episode_history,
                    baseline_ready=baseline.ready,

                    gaze_yaw=gaze_yaw,
                    gaze_pitch=gaze_pitch,
                    gaze_data=gaze_data,
                    behavior_state=behavior_state,
                )

                # ── Landmark dots (key pain AUs) ──────────────────────────
                draw_landmark_dots(
                    frame, patient_landmarks,
                    LEFT_EYE + RIGHT_EYE, color=(80, 255, 80), radius=2
                )
                draw_landmark_dots(
                    frame, patient_landmarks,
                    LEFT_BROW + RIGHT_BROW, color=(80, 80, 255), radius=2
                )

            tracker.draw_box(frame)

        else:
            previous_landmarks = None

        # ── Clinical overlay ──────────────────────────────────────────────
        overlay.render(
            frame=frame,
            au=au,
            ci=ci,
            confidence=confidence,
            active_episodes=ep_detector.active_episodes,

            baseline_ready=baseline.ready,
            baseline_progress=baseline.progress,
            frame_idx=frame_idx,

            behavior_state=behavior_state,
            gaze_yaw=gaze_yaw,
            gaze_pitch=gaze_pitch,
            gaze_data=gaze_data,
        )

        # ── FPS counter ───────────────────────────────────────────────────
        cv2.putText(
            frame,
            f"Frame {frame_idx}",
            (w - 120, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (80, 80, 80), 1
        )

        

        cv2.imshow("Clinical Distress Monitor", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:          # ESC
            break
        elif key == ord('r'):  # 'R' = reset baseline
            baseline = BaselineManager()
            print("[INFO] Baseline reset.")
        elif key == ord('s'):  # 'S' = snapshot
            snap_path = f"outputs/snapshot_{frame_idx:06d}.jpg"
            cv2.imwrite(snap_path, frame)
            print(f"[INFO] Snapshot saved → {snap_path}")

    # ── Cleanup ───────────────────────────────────────────────────────────
    logger.close()
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Session ended.")


if __name__ == "__main__":
    main()
