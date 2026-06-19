"""
temporal/feature_buffer.py
===========================
Multi-window temporal analysis for AU intensities and clinical indices.

Temporal context is essential in clinical pain assessment because:
  - Single-frame scores are noisy (landmark jitter, blinks, speech)
  - Clinically meaningful episodes have duration (typically > 0.5 s)
  - Rate of change distinguishes sudden pain onset from gradual fatigue
  - Rolling variance distinguishes agitation (high) from fatigue (low)
"""

from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional
import numpy as np

from config.settings import CFG


@dataclass
class TemporalStats:
    """Summary statistics for a single signal over a time window."""
    mean:     float = 0.0
    median:   float = 0.0
    ema:      float = 0.0       # exponential moving average
    variance: float = 0.0
    std:      float = 0.0
    peak:     float = 0.0
    rate_of_change: float = 0.0   # (current - mean) / (std + eps)
    n:        int   = 0


class ScalarBuffer:
    """
    Circular buffer for a single scalar signal with multi-window statistics.
    """

    def __init__(self, max_len: int = 1800) -> None:
        self._buf: Deque[float] = deque(maxlen=max_len)
        self._ema: float = 0.0
        self._alpha: float = CFG.temporal.ema_alpha
        self._fps: int = CFG.temporal.fps

    def push(self, value: float) -> None:
        self._buf.append(value)
        if len(self._buf) == 1:
            self._ema = value
        else:
            self._ema = self._alpha * value + (1 - self._alpha) * self._ema

    def stats(self, window_seconds: float = 10.0) -> TemporalStats:
        """
        Compute statistics over the most recent `window_seconds` of data.
        """
        n_frames = max(1, int(window_seconds * self._fps))
        # Extract the window (recent slice)
        buf_list = list(self._buf)
        window = buf_list[-n_frames:]

        if len(window) == 0:
            return TemporalStats()

        arr = np.array(window, dtype=np.float64)
        mean_ = float(np.mean(arr))
        median_ = float(np.median(arr))
        var_  = float(np.var(arr))
        std_  = float(np.std(arr))
        peak_ = float(np.max(arr))
        roc   = float((arr[-1] - mean_) / (std_ + 1e-6))

        return TemporalStats(
            mean=mean_,
            median=median_,
            ema=self._ema,
            variance=var_,
            std=std_,
            peak=peak_,
            rate_of_change=roc,
            n=len(window),
        )

    @property
    def latest(self) -> float:
        return self._buf[-1] if self._buf else 0.0

    @property
    def ema(self) -> float:
        return self._ema

    def detect_peak(
        self,
        window_seconds: float = 10.0,
        threshold_sigma: float = 2.0,
    ) -> bool:
        """Return True if the latest value is `threshold_sigma` above the window mean."""
        s = self.stats(window_seconds)
        if s.n < 5:
            return False
        return (self.latest - s.mean) > threshold_sigma * (s.std + 1e-6)


class TemporalFeatureBuffer:
    """
    Manages per-signal ScalarBuffers for all AU intensities and clinical indices.

    Signals tracked:
        AU intensities:  au4 .. au43
        Clinical indices: pain, fear, fatigue, agitation, tension, respiratory, global
        Derived:          perclos, motion_energy, au26_variance
    """

    _SIGNAL_NAMES = [
        "au4", "au5", "au6", "au7", "au9", "au10",
        "au12", "au15", "au17", "au20", "au23", "au24", "au25", "au26", "au43",
        "pain_index", "fear_index", "fatigue_index", "agitation_index",
        "tension_index", "respiratory_index", "global_distress",
        "perclos", "motion_energy",
    ]

    def __init__(self) -> None:
        cfg = CFG.temporal
        self._buffers: Dict[str, ScalarBuffer] = {
            name: ScalarBuffer(max_len=cfg.buffer_60s)
            for name in self._SIGNAL_NAMES
        }

    def push(self, signal_dict: Dict[str, float]) -> None:
        """Push a dictionary of signal values into the respective buffers."""
        for name, value in signal_dict.items():
            if name in self._buffers:
                self._buffers[name].push(value)

    def get_buffer(self, name: str) -> ScalarBuffer:
        return self._buffers[name]

    def stats(self, name: str, window_seconds: float = 10.0) -> TemporalStats:
        return self._buffers[name].stats(window_seconds)

    def ema(self, name: str) -> float:
        return self._buffers[name].ema

    def latest(self, name: str) -> float:
        return self._buffers[name].latest

    def au26_variance(self, window_seconds: float = 5.0) -> float:
        """
        Rolling variance of AU26 (jaw drop), used as a respiratory proxy.
        Elevated variance indicates cyclical jaw opening typical of laboured breathing.
        """
        s = self.stats("au26", window_seconds)
        return float(np.clip(s.variance * 10.0, 0.0, 1.0))

    def jaw_rhythm_score(self) -> float:
        """
        Crude estimate of jaw movement periodicity ∈ [0, 1].
        High score = regular oscillations (potential respiratory distress marker).
        Uses the 5-second AU26 variance relative to the 30-second mean variance.
        """
        short_var = self.stats("au26", 5.0).variance
        long_mean = self.stats("au26", 30.0).mean
        if long_mean < 1e-6:
            return 0.0
        return float(np.clip(short_var / (long_mean + 1e-6), 0.0, 1.0))

    def smoothed_global(self) -> float:
        """EMA-smoothed global distress score."""
        return self._buffers["global_distress"].ema
