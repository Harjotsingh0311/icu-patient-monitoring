import pandas as pd
from pathlib import Path

INPUT_DIR = Path("datasets/processed")

OUTPUT_FILE = INPUT_DIR / "visual_dataset.csv"

LABEL_MAP = {
    "stable": 0,
    "fatigue": 1,
    "distress": 2,
    "agitation": 3,
}

all_data = []

for csv_file in INPUT_DIR.glob("*_windows.csv"):

    filename = csv_file.stem.lower()

    label = None

    for class_name, class_id in LABEL_MAP.items():

        if filename.startswith(class_name):
            label = class_id
            break

    if label is None:
        print(f"Skipping {csv_file.name}")
        continue

    df = pd.read_csv(csv_file)

    df["label"] = label

    df["source_file"] = csv_file.name

    all_data.append(df)

if len(all_data) == 0:
    raise ValueError(
        "No labeled window files found."
    )

dataset = pd.concat(
    all_data,
    ignore_index=True
)

dataset.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nDataset created successfully")
print(f"Rows: {len(dataset)}")
print(f"Columns: {len(dataset.columns)}")
print(f"Saved to: {OUTPUT_FILE}")

print("\nClass distribution:")
print(dataset["label"].value_counts())