"""
baseline/baseline_manager.py
=============================
Robust patient baseline using median and Median Absolute Deviation (MAD).

Why robust statistics?
-----------------------
Simple mean-based baselines are distorted by early-session distress events
(e.g., a painful procedure during the collection window).  Median and MAD
are resistant to outliers, making the baseline representative of the patient's
true resting facial state even when the collection window is imperfect.

Robust Z-Score
  z = (current - median) / (MAD × 1.4826)
  The scale factor 1.4826 makes MAD consistent with the standard deviation
  of a normal distribution, allowing conventional z-score interpretation.
"""

from __future__ import annotations
import time
from collections import defaultdict
from typing import Dict, List, Optional
import numpy as np

from config.settings import CFG


class BaselineManager:
    """
    Collects resting-state AU observations and builds a robust patient baseline.

    Usage
    -----
    manager = BaselineManager()
    while collecting:
        manager.update(au_frame.as_dict())
    if manager.ready:
        dev = manager.z_score("au4", current_value)
    """

    def __init__(self) -> None:
        self._cfg = CFG.baseline
        self._start_time: float = time.time()
        self._samples: Dict[str, List[float]] = defaultdict(list)
        self._baseline_median: Dict[str, float] = {}
        self._baseline_mad: Dict[str, float] = {}
        self.ready: bool = False

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def update(self, feature_dict: Dict[str, float]) -> None:
        """
        Append one frame of AU observations.
        Automatically finalises the baseline once the collection window elapses.
        """
        if self.ready:
            return

        for key, value in feature_dict.items():
            self._samples[key].append(float(value))

        elapsed = time.time() - self._start_time
        n_samples = len(next(iter(self._samples.values()), []))

        if (elapsed >= self._cfg.collection_seconds and
                n_samples >= self._cfg.min_samples):
            self._compute_baseline()

    def z_score(self, key: str, current: float) -> float:
        """
        Compute the robust Z-score of `current` relative to baseline.

        Returns 0.0 if baseline is not ready or key is unknown.
        Positive → above baseline; Negative → below baseline.
        """
        if not self.ready:
            return 0.0
        med = self._baseline_median.get(key, 0.0)
        mad = self._baseline_mad.get(key, 1.0)
        if mad < 1e-9:
            return 0.0
        return float((current - med) / (mad * self._cfg.mad_scale))

    def deviation(self, key: str, current: float) -> float:
        """
        Absolute deviation from median, normalised by MAD (clamped [-3, 3]).
        Useful when sign does not matter.
        """
        return float(np.clip(self.z_score(key, current), -3.0, 3.0))

    @property
    def progress(self) -> float:
        """Collection progress in [0, 1]."""
        if self.ready:
            return 1.0
        elapsed = time.time() - self._start_time
        n = len(next(iter(self._samples.values()), []))
        time_frac  = min(elapsed / self._cfg.collection_seconds, 1.0)
        count_frac = min(n / max(self._cfg.min_samples, 1), 1.0)
        return (time_frac + count_frac) / 2.0

    @property
    def baseline(self) -> Dict[str, float]:
        """Return the median baseline dictionary."""
        return dict(self._baseline_median)

    # ──────────────────────────────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────────────────────────────

    def _compute_baseline(self) -> None:
        for key, values in self._samples.items():
            arr = np.array(values, dtype=np.float64)
            med = float(np.median(arr))
            mad = float(np.median(np.abs(arr - med)))
            self._baseline_median[key] = med
            self._baseline_mad[key]    = max(mad, 1e-9)
        self.ready = True
