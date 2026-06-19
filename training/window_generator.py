import pandas as pd
from pathlib import Path

WINDOW_SIZE = 150      # 5 sec @ 30 FPS
WINDOW_STEP = 75       # 50% overlap

INPUT_DIR = Path("datasets/raw_logs")
OUTPUT_DIR = Path("datasets/processed")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def process_file(csv_path: Path):

    df = pd.read_csv(csv_path)

    windows = []

    for start in range(0, len(df) - WINDOW_SIZE + 1, WINDOW_STEP):

        window = df.iloc[start:start + WINDOW_SIZE]

        row = {}

        # Clinical Features
        clinical_cols = [
            "global_distress",
            "pain_index",
            "fear_index",
            "fatigue_index",
            "agitation_index",
            "tension_index",
            "respiratory_index",
        ]

        for col in clinical_cols:
            row[f"{col}_mean"] = window[col].mean()
            row[f"{col}_max"] = window[col].max()
            row[f"{col}_std"] = window[col].std()

        # Gaze Features
        gaze_cols = [
            "gaze_variance",
            "fixation_duration",
            "recent_saccades",
            "mean_gaze_speed",
            "max_gaze_speed",
            "eye_contact_ratio",
        ]

        for col in gaze_cols:
            row[f"{col}_mean"] = window[col].mean()
            row[f"{col}_max"] = window[col].max()

        # Confidence
        row["face_confidence_mean"] = window["face_confidence"].mean()
        row["face_confidence_min"] = window["face_confidence"].min()

        # Episode Counts
        episode_cols = [
            "episode_pain",
            "episode_agitation",
            "episode_fatigue",
            "episode_fear",
            "episode_eye_closure",
        ]

        for col in episode_cols:
            row[f"{col}_count"] = window[col].sum()

        # Behavior Ratios
        total = len(window)

        row["alert_ratio"] = (
            (window["behavior_state"] == "alert").sum() / total
        )

        row["scanning_ratio"] = (
            (window["behavior_state"] == "scanning").sum() / total
        )

        row["eye_closed_ratio"] = (
            (window["behavior_state"] == "eye_closed").sum() / total
        )
        row["window_start"] = start
        row["window_end"] = start + WINDOW_SIZE

        windows.append(row)

    output_df = pd.DataFrame(windows)

    out_file = OUTPUT_DIR / f"{csv_path.stem}_windows.csv"

    output_df.to_csv(out_file, index=False)

    print(f"Saved: {out_file}")


def main():

    csv_files = list(INPUT_DIR.glob("*.csv"))

    print(f"Found {len(csv_files)} files")

    for csv_file in csv_files:
        process_file(csv_file)


if __name__ == "__main__":
    main()