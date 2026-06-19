# Clinical Facial Distress Detection System v2

A clinically-inspired real-time facial distress monitoring system for ICU research,
built on Action Unit (AU) based analysis replacing the original handcrafted feature weights.

---

## Folder Structure

```
facial_distress_v2/
├── main.py                        # Entry point — orchestrates full pipeline
├── requirements.txt
├── outputs/                       # Auto-created: CSV + JSONL logs, snapshots
│
├── config/
│   └── settings.py                # ALL constants, thresholds, weights (one place)
│
├── landmarks/
│   ├── landmark_extractor.py      # MediaPipe → typed Landmark objects
│   └── landmark_groups.py         # Named index lists for all facial regions
│
├── action_units/
│   └── au_estimator.py            # Geometric AU estimation (15 AUs)
│
├── clinical_metrics/
│   └── indices.py                 # PSPI pain index + 6 composite sub-indices
│
├── baseline/
│   └── baseline_manager.py        # Robust median/MAD patient baseline
│
├── temporal/
│   └── feature_buffer.py          # Multi-window rolling stats, EMA, peak detection
│
├── episodes/
│   └── episode_detector.py        # State-machine episode detection & logging
│
├── tracking/
│   └── patient_tracker.py         # Multi-face IoU tracking, patient identity lock
│
├── confidence/
│   └── confidence_estimator.py    # Per-frame confidence score (0–1)
│
├── core/
│   ├── head_pose.py               # solvePnP yaw/pitch/roll estimation
│   └── motion.py                  # Motion energy, asymmetry, PERCLOS
│
├── visualization/
│   └── overlay.py                 # Rich clinical overlay with AU bars + episodes
│
└── logging_system/
    └── distress_logger.py         # Per-frame CSV + per-batch JSONL
```

---

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.10+.  No CUDA needed — runs on CPU at 20+ FPS.

---

## Running

```bash
cd facial_distress_v2
python main.py
```

### Keyboard shortcuts

| Key | Action |
|-----|--------|
| ESC | Quit session, flush logs |
| R   | Reset patient baseline |
| S   | Save snapshot to `outputs/` |

---

## What Changed from v1 → v2

### Removed
- Handcrafted weighted score: `EAR×10 + PERCLOS×15 + brow×9 + motion×10 + agitation×3 + MAR×8`
- Simple averaging baseline
- Global score = ad-hoc weighted sum with magic numbers

### Added

| Component | v1 | v2 |
|-----------|----|----|
| Feature basis | EAR, MAR, brow dist, motion energy | 15 Action Units (geometric) |
| Scoring | Manual weights (fixed) | PSPI-inspired clinical formula |
| Baseline | Running mean (30 frames) | Robust median + MAD (30 s) |
| Sub-indices | None | Pain, Fear, Fatigue, Agitation, Tension, Respiratory |
| Temporal analysis | None | 10/30/60 s windows, EMA, peak detection |
| Episodes | None | State-machine: Pain, Agitation, Eye Closure, Fatigue, Fear |
| Multi-face | First face only | IoU patient tracking, identity lock |
| Confidence | None | Per-frame score (landmark quality × pose × tracking × visibility) |
| Logging | None | Per-frame CSV + JSONL |

---

## Output Format

### `outputs/distress_log.csv`

```
timestamp,frame,global_distress,confidence,pain_index,fear_index,fatigue_index,
agitation_index,tension_index,respiratory_index,eye_closure_index,
au4,au5,...,au43,ear_avg,mar,active_episodes,baseline_ready
```

### `outputs/distress_log.jsonl`

One JSON object every 30 frames:
```json
{
  "timestamp": 1749120345.23,
  "frame": 300,
  "au": {"au4": 0.42, "au6": 0.18, ...},
  "clinical": {"pain_index": 45.1, "global_distress": 38.2, ...},
  "confidence": {"overall": 0.91, ...},
  "active_episodes": [],
  "completed_episodes": [{"type": "PAIN", "duration_s": 3.4, ...}],
  "baseline_ready": true
}
```

---

## Clinical Rationale

### Action Unit Basis

**AU4 (Brow Lowerer)** — Corrugator supercilii.
The single most reliable pain indicator (Prkachin, 1992; Pain).
Geometrically estimated as normalised brow-to-upper-eyelid distance.

**AU6 (Cheek Raiser)** — Orbicularis oculi (orbital part).
Raises cheeks, narrows palpebral fissure from below.
Distinguishes genuine pain from voluntary grimacing.

**AU7 (Lid Tightener)** — Orbicularis oculi (palpebral part).
Tightens upper eyelid; characteristic of pain and concentration.

**AU9 (Nose Wrinkler)** — Levator labii superioris alaeque nasi.
Estimated from nostril width relative to IOD.
Present in disgust and pain (Ekman, 1978).

**AU10 (Upper Lip Raiser)** — Levator labii superioris.
Shortens philtrum; accompanies nausea and pain.

**AU43 (Eyes Closed)** — Inverse of Eye Aspect Ratio.
Sustained closure signals pain-driven withdrawal (PSPI component) or fatigue.

### PSPI Formula

```
Pain Index = AU4 + max(AU6, AU7) + max(AU9, AU10) + AU43
```

Normalised to 0–100.  Based on Prkachin & Solomon (2008):
*"The structure, reliability and validity of pain expression."*
Pain Research & Management 13(6):645–655.

### Confidence Score

Uses geometric mean of four sub-factors to strongly penalise any single
failure mode (e.g., extreme head pose alone should halve confidence):

```
Confidence = lm_quality^0.30 × pose_quality^0.30 × tracking^0.25 × visibility^0.15
```

---

## Configuration

All thresholds are in `config/settings.py` — a single frozen dataclass.
No magic numbers elsewhere in the codebase.

To tune the system for a different patient population or camera setup,
edit only `config/settings.py`.
