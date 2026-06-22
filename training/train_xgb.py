import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier


# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

df = pd.read_csv("dataset/processed/visual_dataset.csv")

X = df.drop(columns=["label", "source_file"])
y = df["label"]

# --------------------------------------------------
# Train/Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# --------------------------------------------------
# Model
# --------------------------------------------------

model = XGBClassifier(
    objective="multi:softprob",
    num_class=4,

    n_estimators=100,
    max_depth=4,
    learning_rate=0.05,

    subsample=0.8,
    colsample_bytree=0.8,

    random_state=42
)

model.fit(X_train, y_train)

# --------------------------------------------------
# Evaluate
# --------------------------------------------------

preds = model.predict(X_test)

print("\nAccuracy:")
print(accuracy_score(y_test, preds))

print("\nClassification Report:")
print(classification_report(y_test, preds))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, preds))

# --------------------------------------------------
# Save Model
# --------------------------------------------------

joblib.dump(
    model,
    "trained_models/visual_xgb.pkl"
)

print("\nModel saved successfully")