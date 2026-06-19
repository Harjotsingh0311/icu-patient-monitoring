from collections import deque
import numpy as np


class GazeTracker:

    def __init__(self, window_size=150):

        self.yaw_history = deque(maxlen=window_size)
        self.pitch_history = deque(maxlen=window_size)
        self.fixation_yaw_history = deque(maxlen=30)
        self.fixation_pitch_history = deque(maxlen=30)
        self.eye_contact_frames = 0
        self.valid_gaze_frames = 0

        self.fixation_frames = 0

        self.prev_yaw = None
        self.prev_pitch = None

        self.saccade_count = 0

        # 30 seconds @ 30 FPS
        self.saccade_window = deque(maxlen=150)

        self.gaze_speed_history = deque(
        maxlen=window_size
        )

    def update(self, yaw, pitch):

        if yaw is None:
            return None
        
        self.valid_gaze_frames += 1
        EYE_CONTACT_YAW = 0.25
        EYE_CONTACT_PITCH = 0.25

        if (
            abs(yaw) < EYE_CONTACT_YAW
            and
            abs(pitch) < EYE_CONTACT_PITCH
        ):
            self.eye_contact_frames += 1
        self.yaw_history.append(yaw)
        self.pitch_history.append(pitch)
        
        self.fixation_yaw_history.append(yaw)
        self.fixation_pitch_history.append(pitch)

        yaw_var = np.var(self.yaw_history)
        pitch_var = np.var(self.pitch_history)
        combined_gaze_variance = (
        yaw_var + pitch_var
        )

        fixation_yaw_var = np.var(
            self.fixation_yaw_history
        )

        fixation_pitch_var = np.var(
            self.fixation_pitch_history
        )

        # -----------------------------
        # Saccade Detection
        # -----------------------------

        SACCADE_THRESHOLD = 0.1

        saccade_detected = 0

        if self.prev_yaw is not None:

            yaw_change = abs(yaw - self.prev_yaw)
            pitch_change = abs(pitch - self.prev_pitch)
            gaze_speed = np.sqrt(
            yaw_change**2 +
            pitch_change**2
            )
            self.gaze_speed_history.append(gaze_speed)
        

            if (
                yaw_change > SACCADE_THRESHOLD or
                pitch_change > SACCADE_THRESHOLD
            ):
                self.saccade_count += 1
                saccade_detected = 1
        else:
            self.gaze_speed_history.append(0)

        self.saccade_window.append(
            saccade_detected
        )

        recent_saccades = sum(
            self.saccade_window
        )

       # -----------------------------
        # Fixation Detection
        # -----------------------------

        FIXATION_VAR_THRESHOLD = 0.01
        FIXATION_MOTION_THRESHOLD = 0.08

        if self.prev_yaw is not None:

            yaw_change = abs(yaw - self.prev_yaw)
            pitch_change = abs(pitch - self.prev_pitch)

            if (
                fixation_yaw_var < FIXATION_VAR_THRESHOLD
                and
                fixation_pitch_var < FIXATION_VAR_THRESHOLD
                and
                yaw_change < FIXATION_MOTION_THRESHOLD
                and
                pitch_change < FIXATION_MOTION_THRESHOLD
            ):
                self.fixation_frames += 1
            else:
                self.fixation_frames = 0

        fixation_duration = (
            self.fixation_frames / 30.0
        )

        # -----------------------------
        # Update Previous Values
        # -----------------------------

        self.prev_yaw = yaw
        self.prev_pitch = pitch
        eye_contact_ratio = (
            self.eye_contact_frames /
            self.valid_gaze_frames
        )

        return {
            "yaw_variance": yaw_var,
            "pitch_variance": pitch_var,
            "gaze_variance": combined_gaze_variance,
            "fixation_duration": fixation_duration,
            "saccade_count": self.saccade_count,
            "recent_saccades": recent_saccades,
            "mean_gaze_speed": (
                np.mean(self.gaze_speed_history)
                if len(self.gaze_speed_history) > 0
                else 0
            ),

            "max_gaze_speed": (
                np.max(self.gaze_speed_history)
                if len(self.gaze_speed_history) > 0
                else 0
            ),
            "eye_contact_ratio": eye_contact_ratio,
            
        }