"""
episodes/episode_detector.py
=============================
Detects, tracks, and records discrete distress episodes.

An episode has:
  - Type: PAIN | AGITATION | EYE_CLOSURE | FATIGUE | FEAR
  - Start frame / timestamp
  - Peak severity within the episode
  - End frame / timestamp (once the score drops below threshold)
  - Duration
  - Mean severity during episode

Episodes require `min_episode_frames` of sustained elevated score before
being confirmed, to avoid false positives from transient landmark noise.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional

from config.settings import CFG


class EpisodeType(Enum):
    PAIN        = auto()
    AGITATION   = auto()
    EYE_CLOSURE = auto()
    FATIGUE     = auto()
    FEAR        = auto()


@dataclass
class Episode:
    """A single detected distress episode."""
    episode_type:  EpisodeType
    start_time:    float           # Unix timestamp
    start_frame:   int
    peak_severity: float = 0.0
    end_time:      Optional[float] = None
    end_frame:     Optional[int]   = None
    severity_sum:  float = 0.0
    frame_count:   int   = 0

    @property
    def active(self) -> bool:
        return self.end_time is None

    @property
    def duration_seconds(self) -> float:
        t_end = self.end_time or time.time()
        return t_end - self.start_time

    @property
    def mean_severity(self) -> float:
        if self.frame_count == 0:
            return 0.0
        return self.severity_sum / self.frame_count

    def as_dict(self) -> dict:
        return {
            "type":           self.episode_type.name,
            "start_time":     self.start_time,
            "start_frame":    self.start_frame,
            "end_time":       self.end_time,
            "end_frame":      self.end_frame,
            "peak_severity":  round(self.peak_severity, 2),
            "mean_severity":  round(self.mean_severity, 2),
            "duration_s":     round(self.duration_seconds, 2),
        }


@dataclass
class _PendingEpisode:
    """Candidate episode waiting to be confirmed."""
    episode_type:  EpisodeType
    start_time:    float
    start_frame:   int
    frames_active: int = 0
    peak_so_far:   float = 0.0


class EpisodeDetector:
    """
    State machine that tracks active and candidate episodes for all types.

    Usage
    -----
    detector = EpisodeDetector()
    # each frame:
    detector.update(
        frame_idx=i,
        pain_score=ci.pain_index,
        agitation_score=ci.agitation_index,
        eye_closure_score=ci.eye_closure_index,
        fatigue_score=ci.fatigue_index,
        fear_score=ci.fear_index,
    )
    active = detector.active_episodes
    history = detector.episode_history
    """

    def __init__(self) -> None:
        self._cfg = CFG.episode
        self._frame: int = 0
        # Currently confirmed active episodes (one per type max)
        self._active: dict[EpisodeType, Episode] = {}
        # Candidate episodes waiting for confirmation
        self._pending: dict[EpisodeType, _PendingEpisode] = {}
        # Cooldown counters per type (frames until new episode allowed)
        self._cooldown: dict[EpisodeType, int] = {}
        # Completed episode history
        self._history: List[Episode] = []

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def update(
        self,
        frame_idx: int,
        pain_score: float,
        agitation_score: float,
        eye_closure_score: float,
        fatigue_score: float,
        fear_score: float,
    ) -> None:
        """Process one frame of scores."""
        self._frame = frame_idx
        now = time.time()

        scores = {
            EpisodeType.PAIN:        (pain_score,        self._cfg.pain_threshold),
            EpisodeType.AGITATION:   (agitation_score,   self._cfg.agitation_threshold),
            EpisodeType.EYE_CLOSURE: (eye_closure_score, self._cfg.eye_closure_threshold * 100),
            EpisodeType.FATIGUE:     (fatigue_score,     self._cfg.fatigue_threshold),
            EpisodeType.FEAR:        (fear_score,        self._cfg.pain_threshold),   # same threshold
        }

        for ep_type, (score, threshold) in scores.items():
            self._update_type(ep_type, score, threshold, now)

        # Decrement cooldowns
        for ep_type in list(self._cooldown):
            self._cooldown[ep_type] -= 1
            if self._cooldown[ep_type] <= 0:
                del self._cooldown[ep_type]

    @property
    def active_episodes(self) -> List[Episode]:
        return list(self._active.values())

    @property
    def episode_history(self) -> List[Episode]:
        return list(self._history)

    def active_episode_names(self) -> List[str]:
        return [ep.episode_type.name for ep in self._active.values()]

    # ──────────────────────────────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────────────────────────────

    def _update_type(
        self,
        ep_type: EpisodeType,
        score: float,
        threshold: float,
        now: float,
    ) -> None:
        in_cooldown = ep_type in self._cooldown
        above_threshold = score >= threshold

        if ep_type in self._active:
            ep = self._active[ep_type]
            ep.frame_count += 1
            ep.severity_sum += score
            ep.peak_severity = max(ep.peak_severity, score)
            # End episode if score drops below threshold
            if not above_threshold:
                ep.end_time  = now
                ep.end_frame = self._frame
                self._history.append(ep)
                del self._active[ep_type]
                self._cooldown[ep_type] = self._cfg.cooldown_frames

        elif ep_type in self._pending:
            pend = self._pending[ep_type]
            if above_threshold:
                pend.frames_active += 1
                pend.peak_so_far = max(pend.peak_so_far, score)
                if pend.frames_active >= self._cfg.min_episode_frames:
                    # Confirm episode
                    self._active[ep_type] = Episode(
                        episode_type=ep_type,
                        start_time=pend.start_time,
                        start_frame=pend.start_frame,
                        peak_severity=pend.peak_so_far,
                        severity_sum=score,
                        frame_count=1,
                    )
                    del self._pending[ep_type]
            else:
                # Candidate failed confirmation
                del self._pending[ep_type]

        else:
            # Start tracking a candidate
            if above_threshold and not in_cooldown:
                self._pending[ep_type] = _PendingEpisode(
                    episode_type=ep_type,
                    start_time=now,
                    start_frame=self._frame,
                    frames_active=1,
                    peak_so_far=score,
                )
