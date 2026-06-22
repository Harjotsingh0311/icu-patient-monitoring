"""
clinical_metrics/indices.py
============================
Computation of all clinical distress sub-indices and the Global Distress Index.

Clinical basis
--------------
Pain Index (PI)
  Implements the Prkachin–Solomon Pain Intensity (PSPI) formula:
    PSPI = AU4 + max(AU6, AU7) + max(AU9, AU10) + AU43
  (Prkachin & Solomon, 2008; Pain Research & Management)
  Normalised to [0, 100].

Fear/Anxiety Index
  Combines AUs associated with the fear/anxiety expression cluster
  (Ekman 1978): AU5 (wide eyes), AU20 (lip stretch), AU26 (jaw drop),
  plus bilateral symmetry break which occurs under acute threat.

Fatigue Index
  PERCLOS (Percentage of Eye Closure) is the gold standard drowsiness
  indicator (Dinges & Mallis, 1998).  Combined with AU43 and low facial
  movement (indicating reduced alertness).

Agitation Index
  Elevated motion energy + rapid jaw movement + head pose deviation.
  Validated in ICU delirium research (Bergeron et al, 2001).

Facial Tension Index
  AU4 + AU7 + AU23 + AU24 — muscles that tense without opening the mouth,
  characteristic of stoic pain, anxiety, or nausea.

Respiratory Distress Proxy
  AU26 (jaw drop) variance + AU9 (nostril flare) + jaw movement rhythm.
  Increased respiratory effort often produces synchronised facial motion
  visible in ICU patients on supplemental oxygen.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from action_units.au_estimator import AUFrame
from config.settings import CFG


@dataclass
class ClinicalIndices:
    """All computed clinical sub-indices and the global score."""
    pain_index:        float = 0.0   # PSPI-inspired, 0-100
    fear_index:        float = 0.0   # Fear/Anxiety, 0-100
    fatigue_index:     float = 0.0   # Fatigue, 0-100
    agitation_index:   float = 0.0   # Agitation, 0-100
    tension_index:     float = 0.0   # Facial Tension, 0-100
    respiratory_index: float = 0.0   # Respiratory Distress Proxy, 0-100
    eye_closure_index: float = 0.0   # PERCLOS-based, 0-100
    global_distress:   float = 0.0   # Weighted aggregate, 0-100

    def severity_label(self) -> str:
        """Map global_distress to a human-readable severity tier."""
        s = self.global_distress
        if s < 20:
            return "Normal"
        elif s < 40:
            return "Mild"
        elif s < 60:
            return "Moderate"
        elif s < 80:
            return "Severe"
        else:
            return "Critical"

    def as_dict(self) -> dict:
        return {k: round(v, 2) for k, v in self.__dict__.items()}


def compute_clinical_indices(
    au: AUFrame,
    perclos: float,
    motion_energy: float,
    head_pose_deviation: float,
    asymmetry: float,
    au26_variance: float,
    jaw_rhythm: float,
) -> ClinicalIndices:
    """
    Compute all clinical indices from Action Unit intensities and derived signals.

    Parameters
    ----------
    au                 : AUFrame    – current frame AU intensities [0,1]
    perclos            : float      – rolling eye-closure fraction [0,1]
    motion_energy      : float      – normalised facial motion [0,1]
    head_pose_deviation: float      – combined yaw+pitch+roll deviation [0,1]
    asymmetry          : float      – facial asymmetry score [0,1]
    au26_variance      : float      – rolling variance of AU26 [0,1]
    jaw_rhythm         : float      – periodicity of jaw movement [0,1]

    Returns
    -------
    ClinicalIndices
    """
    cfg_pain = CFG.pain_index
    cfg_clin = CFG.clinical

    ci = ClinicalIndices()

    # ── Pain Index (PSPI) ────────────────────────────────────────────────
    # Formula: AU4 + max(AU6, AU7) + max(AU9, AU10) + AU43
    raw_pain = (
        cfg_pain.au4_weight     * au.au4 +
        cfg_pain.au6_au7_weight * max(au.au6, au.au7) +
        cfg_pain.au9_au10_weight * max(au.au9, au.au10) +
        cfg_pain.au43_weight    * au.au43
    )
    print(
    f"AU4={au.au4:.2f} "
    f"AU6={au.au6:.2f} "
    f"AU7={au.au7:.2f} "
    f"AU9={au.au9:.2f} "
    f"AU10={au.au10:.2f} "
    f"AU43={au.au43:.2f} "
    f"RAW={raw_pain:.2f}"
)
    ci.pain_index = float(np.clip(raw_pain / cfg_pain.max_raw * 100.0, 0.0, 100.0))

    # ── Fear / Anxiety Index ─────────────────────────────────────────────
    fw = cfg_clin.fear_weights
    raw_fear = (
        fw["au5"]      * au.au5  +
        fw["au20"]     * au.au20 +
        fw["au26"]     * au.au26 +
        fw["asymmetry"] * asymmetry
    )
    ci.fear_index = float(np.clip(raw_fear * 100.0, 0.0, 100.0))

    # ── Fatigue Index ────────────────────────────────────────────────────
    fatw = cfg_clin.fatigue_weights
    low_variance = float(np.clip(1.0 - motion_energy, 0.0, 1.0))  # stillness
    raw_fatigue = (
        fatw["au43"]        * au.au43 +
        fatw["perclos"]     * perclos +
        fatw["low_variance"] * low_variance
    )
    ci.fatigue_index = float(np.clip(raw_fatigue * 100.0, 0.0, 100.0))

    # ── Eye Closure Index ────────────────────────────────────────────────
    # Simple PERCLOS scaled 0-100
    ci.eye_closure_index = float(np.clip(perclos * 100.0, 0.0, 100.0))

    # ── Agitation Index ──────────────────────────────────────────────────
    agw = cfg_clin.agitation_weights
    raw_agit = (
        agw["motion"]    * motion_energy +
        agw["au26"]      * au.au26 +
        agw["head_pose"] * head_pose_deviation
    )
    ci.agitation_index = float(np.clip(raw_agit * 100.0, 0.0, 100.0))

    # ── Facial Tension Index ─────────────────────────────────────────────
    tw = cfg_clin.tension_weights
    raw_tension = (
        tw["au4"]  * au.au4  +
        tw["au7"]  * au.au7  +
        tw["au23"] * au.au23 +
        tw["au24"] * au.au24
    )
    ci.tension_index = float(np.clip(raw_tension * 100.0, 0.0, 100.0))

    # ── Respiratory Distress Proxy ────────────────────────────────────────
    rw = cfg_clin.respiratory_weights
    raw_resp = (
        rw["au26_variance"] * au26_variance +
        rw["au9"]           * au.au9 +
        rw["jaw_rhythm"]    * jaw_rhythm
    )
    ci.respiratory_index = float(np.clip(raw_resp * 100.0, 0.0, 100.0))

    # ── Global Distress Index ─────────────────────────────────────────────
    gw = cfg_clin.global_weights
    raw_global = (
        gw["pain_index"]        * ci.pain_index +
        gw["fear_index"]        * ci.fear_index +
        gw["fatigue_index"]     * ci.fatigue_index +
        gw["agitation_index"]   * ci.agitation_index +
        gw["tension_index"]     * ci.tension_index +
        gw["respiratory_index"] * ci.respiratory_index
    )
    ci.global_distress = float(np.clip(raw_global, 0.0, 100.0))

    return ci
