"""
clinical_metrics/indices.py
============================
Computation of all clinical distress sub-indices and the Global Distress Index.

CALIBRATION REVISION — see CALIBRATION_REPORT.md for rationale.

Key fix vs. original:
  raw_pain computation: removed the spurious `+ 1.2 * au.au25 + 1.5 * au.au26`
  terms that were appended to the PSPI formula.  Those addends:
    (a) are not part of the Prkachin–Solomon formula,
    (b) inflated max_raw unpredictably (they were not accounted for in
        PainIndexConfig.max_raw=9.5), producing a compressed pain scale
        that masked moderate pain events, and
    (c) were placed on a line that ran past the `print()` debug statement
        without a separator, making them invisible in code review.
  AU25 and AU26 are still captured in agitation_index and respiratory_index
  where they clinically belong.

  Also: the debug print() block after `return ci` is unreachable — removed.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from action_units.au_estimator import AUFrame
from config.settings import CFG


@dataclass
class ClinicalIndices:
    """All computed clinical sub-indices and the global score."""
    pain_index:        float = 0.0
    fear_index:        float = 0.0
    fatigue_index:     float = 0.0
    agitation_index:   float = 0.0
    tension_index:     float = 0.0
    respiratory_index: float = 0.0
    eye_closure_index: float = 0.0
    global_distress:   float = 0.0

    def severity_label(self) -> str:
        s = self.global_distress
        if s < 20:   return "Normal"
        elif s < 40: return "Mild"
        elif s < 60: return "Moderate"
        elif s < 80: return "Severe"
        else:        return "Critical"

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
    baseline=None,
) -> ClinicalIndices:
    cfg_pain = CFG.pain_index
    cfg_clin = CFG.clinical

    ci = ClinicalIndices()

    # ── Pain Index (PSPI) ────────────────────────────────────────────────
    # Formula: AU4 + max(AU6, AU7) + max(AU9, AU10) + AU43
    # (Prkachin & Solomon, 2008)
    au4_val   = au.au4
    au67_val  = max(au.au6, au.au7)
    au910_val = max(au.au9, au.au10)
    au43_val  = au.au43

    if baseline is not None:
        au4_delta = max(0.0, au.au4 - baseline.get("au4", 0.0))
        au67_delta = max(
            0.0,
            max(au.au6, au.au7) - max(baseline.get("au6", 0.0), baseline.get("au7", 0.0))
        )
        au910_delta = max(
            0.0,
            max(au.au9, au.au10) - max(baseline.get("au9", 0.0), baseline.get("au10", 0.0))
        )
        au43_delta = max(0.0, au.au43 - baseline.get("au43", 0.0))

        au4_val   = 0.7 * au.au4              + 0.3 * au4_delta
        au67_val  = 0.7 * max(au.au6, au.au7) + 0.3 * au67_delta
        au910_val = 0.7 * max(au.au9, au.au10)+ 0.3 * au910_delta
        au43_val  = 0.7 * au.au43             + 0.3 * au43_delta

    # ── PSPI formula — pure, no extra addends ────────────────────────────
    # ORIGINAL BUG: the line below had `+ 1.2 * au.au25 + 1.5 * au.au26`
    # appended (continuation of the same expression).  These terms were NOT
    # part of the PSPI formula and were not reflected in max_raw=9.5,
    # causing systematic score compression.  Removed here.
    raw_pain = (
        cfg_pain.au4_weight      * au4_val   +
        cfg_pain.au6_au7_weight  * au67_val  +
        cfg_pain.au9_au10_weight * au910_val +
        cfg_pain.au43_weight     * au43_val
    )

    ci.pain_index = float(np.clip(raw_pain / cfg_pain.max_raw * 100.0, 0.0, 100.0))
    

    # ── Fear / Anxiety Index ─────────────────────────────────────────────
    fw = cfg_clin.fear_weights
    raw_fear = (
        fw["au5"]       * au.au5   +
        fw["au20"]      * au.au20  +
        fw["au26"]      * au.au26  +
        fw["asymmetry"] * asymmetry
    )
    ci.fear_index = float(np.clip(raw_fear * 100.0, 0.0, 100.0))

    # ── Fatigue Index ────────────────────────────────────────────────────
    fatw = cfg_clin.fatigue_weights
    low_variance = float(np.clip(1.0 - motion_energy, 0.0, 1.0))
    raw_fatigue = (
        fatw["au43"]         * au.au43   +
        fatw["perclos"]      * perclos   +
        fatw["low_variance"] * low_variance
    )
    ci.fatigue_index = float(np.clip(raw_fatigue * 100.0, 0.0, 100.0))

    # ── Eye Closure Index ────────────────────────────────────────────────
    ci.eye_closure_index = float(np.clip(perclos * 100.0, 0.0, 100.0))

    # ── Agitation Index ──────────────────────────────────────────────────
    agw = cfg_clin.agitation_weights
    raw_agit = (
        agw["motion"]    * motion_energy       +
        agw["au26"]      * au.au26             +
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
        rw["au9"]           * au.au9        +
        rw["jaw_rhythm"]    * jaw_rhythm
    )
    ci.respiratory_index = float(np.clip(raw_resp * 100.0, 0.0, 100.0))

    # ── Global Distress Index ─────────────────────────────────────────────
    gw = cfg_clin.global_weights
    raw_global = (
        gw["pain_index"]        * ci.pain_index        +
        gw["fear_index"]        * ci.fear_index        +
        gw["fatigue_index"]     * ci.fatigue_index     +
        gw["agitation_index"]   * ci.agitation_index   +
        gw["tension_index"]     * ci.tension_index     +
        gw["respiratory_index"] * ci.respiratory_index
    )
    ci.global_distress = float(np.clip(raw_global, 0.0, 100.0))

    return ci