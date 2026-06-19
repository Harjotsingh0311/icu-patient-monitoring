"""
visualization/overlay.py
=========================
Real-time clinical overlay for the distress detection system.

Layout
------
Left panel (320 px):
  ┌─────────────────────────────┐
  │ PATIENT STATUS              │
  │ Global Distress   ████ 68.2 │
  │ Confidence            0.92  │
  │ ─── Clinical Indices ────── │
  │ Pain Index        ████ 45.1 │
  │ Fear/Anxiety      █░░░ 22.0 │
  │ Fatigue           ██░░ 38.4 │
  │ Agitation         █░░░ 18.2 │
  │ Tension           ███░ 51.0 │
  │ Respiratory       █░░░ 12.3 │
  │ ─── Action Units ─────────  │
  │ AU4  Brow Lower   ██░░ 0.42 │
  │ AU6  Cheek Raise  █░░░ 0.18 │
  │ AU9  Nose Wrinkle ░░░░ 0.05 │
  │ AU43 Eye Closure  ███░ 0.60 │
  │ ─── Active Episodes ─────── │
  │ ⚠ PAIN  [00:12]            │
  │ Baseline: READY             │
  └─────────────────────────────┘
"""

from __future__ import annotations
import cv2
import numpy as np
from typing import List

from action_units.au_estimator import AUFrame
from clinical_metrics.indices import ClinicalIndices
from confidence.confidence_estimator import ConfidenceResult
from episodes.episode_detector import Episode
from config.settings import CFG


class DistressOverlay:
    """Renders the full clinical overlay onto a frame."""

    def __init__(self) -> None:
        self._cfg = CFG.visualization
        self._font = cv2.FONT_HERSHEY_SIMPLEX

    # ──────────────────────────────────────────────────────────────────────
    # Public entry point
    # ──────────────────────────────────────────────────────────────────────

    def render(
        self,
        frame: np.ndarray,
        au: AUFrame,
        ci: ClinicalIndices,
        confidence: ConfidenceResult,
        active_episodes: List[Episode],
        baseline_ready: bool,
        baseline_progress: float,
        frame_idx: int,
    ) -> None:
        """Draw the full overlay onto `frame` in-place."""
        c = self._cfg
        panel_w = c.panel_width
        panel_h = min(frame.shape[0], 660)

        # Semi-transparent background panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (8, 8), (panel_w, panel_h), c.color_panel_bg, -1)
        cv2.addWeighted(overlay, c.panel_alpha, frame, 1 - c.panel_alpha, 0, frame)

        y = 26
        lh = c.line_height

        # ── Header ──────────────────────────────────────────────────────
        self._text(frame, "CLINICAL DISTRESS MONITOR", 14, y, (200, 200, 200), scale=0.45)
        y += lh

        # ── Global score + severity colour ───────────────────────────────
        sev_color = self._severity_color(ci.global_distress)
        self._bar_row(frame, "Global Distress", ci.global_distress / 100.0,
                      y, sev_color, val_suffix=f"{ci.global_distress:.1f}")
        y += lh

        label = ci.severity_label()
        self._text(frame, f"Level: {label}", 14, y, sev_color, scale=0.44)
        y += lh - 2

        conf_color = self._confidence_color(confidence.overall)
        self._text(frame,
                   f"Confidence: {confidence.overall:.2f}",
                   14, y, conf_color, scale=0.43)
        y += lh + 4

        # ── Clinical Indices ─────────────────────────────────────────────
        self._section(frame, "CLINICAL INDICES", y)
        y += lh

        for label, val in [
            ("Pain Index",    ci.pain_index),
            ("Fear/Anxiety",  ci.fear_index),
            ("Fatigue",       ci.fatigue_index),
            ("Agitation",     ci.agitation_index),
            ("Tension",       ci.tension_index),
            ("Respiratory",   ci.respiratory_index),
            ("Eye Closure",   ci.eye_closure_index),
        ]:
            color = self._severity_color(val)
            self._bar_row(frame, label, val / 100.0, y, color, val_suffix=f"{val:.1f}")
            y += lh

        y += 4

        # ── Key Action Units ─────────────────────────────────────────────
        self._section(frame, "ACTION UNITS (KEY)", y)
        y += lh

        for au_label, au_val in [
            ("AU4  Brow Lower",    au.au4),
            ("AU6  Cheek Raise",   au.au6),
            ("AU7  Lid Tighten",   au.au7),
            ("AU9  Nose Wrinkle",  au.au9),
            ("AU43 Eye Close",     au.au43),
            ("AU25 Lips Part",     au.au25),
            ("AU26 Jaw Drop",      au.au26),
        ]:
            color = self._au_color(au_val)
            self._bar_row(frame, au_label, au_val, y, color, val_suffix=f"{au_val:.2f}")
            y += lh

        y += 4

        # ── Active Episodes ───────────────────────────────────────────────
        self._section(frame, "ACTIVE EPISODES", y)
        y += lh

        if active_episodes:
            for ep in active_episodes:
                dur = ep.duration_seconds
                ep_text = f"! {ep.episode_type.name}  {dur:.0f}s  pk:{ep.peak_severity:.0f}"
                self._text(frame, ep_text, 14, y, self._cfg.color_episode, scale=0.42)
                y += lh
        else:
            self._text(frame, "None", 14, y, (100, 200, 100), scale=0.42)
            y += lh

        y += 4

        # ── Baseline status ───────────────────────────────────────────────
        if baseline_ready:
            bline_text = "Baseline: READY"
            bline_col  = (80, 220, 80)
        else:
            pct = int(baseline_progress * 100)
            bline_text = f"Baseline: {pct}% ..."
            bline_col  = (60, 200, 255)
        self._text(frame, bline_text, 14, y, bline_col, scale=0.43)
        y += lh

        # ── Frame counter (bottom of panel) ──────────────────────────────
        self._text(frame, f"Frame #{frame_idx}", 14, y, (90, 90, 90), scale=0.38)

    # ──────────────────────────────────────────────────────────────────────
    # Drawing helpers
    # ──────────────────────────────────────────────────────────────────────

    def _text(
        self,
        frame: np.ndarray,
        text: str,
        x: int,
        y: int,
        color: tuple,
        scale: float | None = None,
    ) -> None:
        s = scale or self._cfg.font_scale
        cv2.putText(frame, text, (x, y), self._font, s, color,
                    self._cfg.font_thickness, cv2.LINE_AA)

    def _section(self, frame: np.ndarray, title: str, y: int) -> None:
        cv2.line(frame, (10, y - 4), (self._cfg.panel_width - 4, y - 4),
                 (60, 60, 80), 1)
        self._text(frame, title, 14, y, self._cfg.color_label, scale=0.40)

    def _bar_row(
        self,
        frame: np.ndarray,
        label: str,
        fraction: float,
        y: int,
        color: tuple,
        val_suffix: str = "",
    ) -> None:
        """Draw a label + filled progress bar + numeric value."""
        c = self._cfg
        label_x = 14
        bar_x   = 152
        bar_y   = y - 9
        bar_w   = c.bar_width
        bar_h   = c.bar_height
        val_x   = bar_x + bar_w + 4

        # Label
        self._text(frame, label, label_x, y, c.color_text, scale=0.40)

        # Background bar
        cv2.rectangle(frame,
                      (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h),
                      (45, 45, 55), -1)

        # Filled bar
        fill = int(np.clip(fraction, 0.0, 1.0) * bar_w)
        if fill > 0:
            cv2.rectangle(frame,
                          (bar_x, bar_y),
                          (bar_x + fill, bar_y + bar_h),
                          color, -1)

        # Numeric value
        self._text(frame, val_suffix, val_x, y, c.color_text, scale=0.38)

    def _severity_color(self, score: float) -> tuple:
        """BGR colour based on score 0-100."""
        if score < 20:
            return self._cfg.color_ok
        elif score < 40:
            return self._cfg.color_mild
        elif score < 60:
            return self._cfg.color_moderate
        else:
            return self._cfg.color_severe

    def _confidence_color(self, conf: float) -> tuple:
        if conf >= 0.75:
            return (80, 220, 80)
        elif conf >= 0.50:
            return (60, 200, 255)
        else:
            return (40, 80, 255)

    def _au_color(self, intensity: float) -> tuple:
        """Blue→yellow gradient for AU bars."""
        r = int(np.clip(intensity * 2 * 255, 0, 255))
        g = int(np.clip(intensity * 255, 0, 255))
        b = int(np.clip((1 - intensity) * 255, 0, 255))
        return (b, g, r)


def draw_landmark_dots(
    frame: np.ndarray,
    landmarks,
    indices: list,
    color: tuple = (0, 255, 100),
    radius: int = 2,
) -> None:
    """Draw small circles at selected landmark positions."""
    for i in indices:
        lm = landmarks[i]
        cv2.circle(frame, (int(lm.x), int(lm.y)), radius, color, -1)
