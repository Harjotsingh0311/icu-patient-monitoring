# ICU Multimodal Patient Monitoring System

Real-time multimodal patient monitoring system for ICU and clinical research environments.

The system combines:

* Facial Action Unit (AU) analysis
* Clinical distress indices
* Head pose estimation
* Motion analysis
* Eye gaze estimation (L2CS-Net)
* Fixation and saccade analysis
* Eye-contact estimation
* PERCLOS-based vigilance monitoring
* Temporal episode detection

into a unified fusion pipeline capable of generating clinically relevant behavioral and distress indicators from a single RGB camera.

---

# System Architecture

```text
Video Input
     │
     ▼
MediaPipe Face Mesh
     │
 ┌───┴───────────┐
 │               │
 ▼               ▼
Face Pipeline   Eye Pipeline
 │               │
 ▼               ▼
Clinical      Gaze Features
Features
 │               │
 └──────┬────────┘
        ▼
  Fusion Layer
        ▼
 Clinical Log
        ▼
 CSV / JSONL Output
```

---

# Project Structure

```text
facial_distress_v2/
│
├── main.py
├── requirements.txt
│
├── action_units/
│   └── au_estimator.py
│
├── baseline/
│   └── baseline_manager.py
│
├── clinical_metrics/
│   └── indices.py
│
├── confidence/
│   └── confidence_estimator.py
│
├── config/
│   └── settings.py
│
├── core/
│   ├── head_pose.py
│   └── motion.py
│
├── episodes/
│   └── episode_detector.py
│
├── eye/
│   ├── gaze_estimator.py
│   └── gaze_tracker.py
│
├── l2cs/
│   ├── model.py
│   ├── pipeline.py
│   ├── results.py
│   └── utils.py
│
├── landmarks/
│   ├── landmark_extractor.py
│   └── landmark_groups.py
│
├── logging_system/
│   └── distress_logger.py
│
├── models/
│   └── L2CSNet_gaze360.pkl
│
├── temporal/
│   └── feature_buffer.py
│
├── tracking/
│   └── patient_tracker.py
│
├── visualization/
│   └── overlay.py
│
└── outputs/
```

---

# Features

## Facial Analysis

### Action Units

The system estimates:

* AU4 – Brow Lowerer
* AU5 – Upper Lid Raiser
* AU6 – Cheek Raiser
* AU7 – Lid Tightener
* AU9 – Nose Wrinkler
* AU10 – Upper Lip Raiser
* AU12 – Lip Corner Puller
* AU15 – Lip Corner Depressor
* AU17 – Chin Raiser
* AU20 – Lip Stretcher
* AU23 – Lip Tightener
* AU24 – Lip Pressor
* AU25 – Lips Part
* AU26 – Jaw Drop
* AU43 – Eye Closure

---

### Clinical Indices

Computed continuously:

* Pain Index
* Fear Index
* Fatigue Index
* Agitation Index
* Tension Index
* Respiratory Distress Index
* Global Distress Score

---

### Head Pose

* Head Yaw
* Head Pitch
* Head Roll
* Pose Deviation

---

### Motion Features

* Motion Energy
* Facial Asymmetry
* Temporal Variability

---

## Eye Analysis

The eye pipeline uses L2CS-Net gaze estimation integrated into the facial pipeline.

### Gaze Features

* Gaze Yaw
* Gaze Pitch
* Gaze Variance

### Eye Behavior Features

* Fixation Duration
* Recent Saccades
* Mean Gaze Speed
* Maximum Gaze Speed
* Eye Contact Ratio

### Vigilance Features

* Blink Detection
* Eye Closure Detection
* PERCLOS

### Behavioral States

The system automatically classifies:

* Alert
* Eye Closed
* Fixation
* Scanning
* Drowsy

---

## Temporal Analysis

The temporal engine maintains rolling windows and computes:

* Exponential Moving Averages
* Variance Metrics
* Peak Detection
* Episode Detection

Detected episodes include:

* Pain
* Agitation
* Fatigue
* Fear
* Eye Closure

---

# Installation

## 1. Clone the Repository

```bash
git clone <repository-url>
cd facial_distress_v2
```

---

## 2. Create a Python Environment (Recommended)

```bash
conda create -n gaze python=3.10
conda activate gaze
```

Alternatively, you may use a standard Python virtual environment.

---

## 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Install L2CS-Net

This project uses a locally bundled copy of **L2CS-Net** for appearance-based gaze estimation.

Install the package in editable mode from the repository root:

```bash
pip install -e ./L2CS-Net
```

or

```bash
cd L2CS-Net
pip install -e .
cd ..
```

This makes the `l2cs` package available to the main pipeline while allowing local modifications without reinstalling.

---

## 5. Download the L2CS Pretrained Model

Download the pretrained **Gaze360** weights (`L2CSNet_gaze360.pkl`) and place them inside:

```text
models/
└── L2CSNet_gaze360.pkl
```

Update the model path in the project configuration if the weights are stored in a different location.

---

## 6. Verify Installation

The following command should execute without import errors:

```bash
python main.py
```

If successful, the application will initialize:

* MediaPipe Face Mesh
* L2CS-Net Gaze Estimator
* Clinical Feature Extraction
* Behavioral Analysis
* Temporal Episode Detection

and begin real-time monitoring from the default camera.

---

### System Requirements

* Python 3.10+
* Windows/Linux
* CUDA-capable GPU (optional, recommended for higher FPS)
* Webcam or RGB Camera

---


# Output

## CSV Log

```text
outputs/distress_log.csv
```

Contains:

### Facial Features

* AU Intensities
* Clinical Indices
* Confidence Scores
* Episode Flags

### Eye Features

* gaze_yaw
* gaze_pitch
* gaze_variance
* fixation_duration
* recent_saccades
* mean_gaze_speed
* max_gaze_speed
* eye_contact_ratio
* behavior_state

---

## JSONL Log

```text
outputs/distress_log.jsonl
```

Stores structured snapshots of:

* AU values
* Clinical metrics
* Confidence metrics
* Active episodes
* Completed episodes

---

# Current Fusion Schema

The final fusion model currently combines:

```text
Facial Features
+
Clinical Features
+
Head Pose Features
+
Eye Gaze Features
+
Behavioral Features
+
Temporal Episode Features
```

into a single multimodal representation.

---

# Research Applications

* ICU Patient Monitoring
* Non-Verbal Pain Assessment
* Sedation Monitoring
* Fatigue Detection
* Vigilance Monitoring
* Human Behavior Analysis
* Clinical Decision Support

---

# Future Work

Planned extensions:

* Audio Distress Analysis
* Multimodal Fusion Model Training
* XGBoost-Based Patient State Classification
* Temporal Transformer Models
* 3D Stereo Vision Integration
* Bed Fall Risk Detection
* Web Dashboard
* Mobile Application
* Jetson Deployment

---

# License

Research and educational use.
