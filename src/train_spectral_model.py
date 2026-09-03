from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# PATHS
# --------------------------------------------------

DATA_PATH = Path("data/spectral/spectral_features.csv")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "spectral_rf.pkl"


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("Loading spectral dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Samples: {len(df)}")


# --------------------------------------------------
# FEATURES / LABEL
# --------------------------------------------------

features = [
    "intensity",
    "local_mean",
    "local_std",
    "local_contrast",
]

X = df[features]
y = df["target"]


# --------------------------------------------------
# TRAIN / VALIDATION SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print(f"Training samples:   {len(X_train)}")
print(f"Validation samples: {len(X_test)}")


# --------------------------------------------------
# RANDOM FOREST
# --------------------------------------------------

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced",
)

model.fit(X_train, y_train)


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

print("Evaluating model...")

predictions = model.predict(X_test)


# --------------------------------------------------
# METRICS
# --------------------------------------------------

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n")
print("=" * 55)
print("SPECTRAL ML MODEL RESULTS")
print("=" * 55)

print(
    f"\nAccuracy: {accuracy:.4f}"
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "Background",
            "Target"
        ]
    )
)

print("Confusion Matrix:")
print(
    confusion_matrix(
        y_test,
        predictions
    )
)


# --------------------------------------------------
# FEATURE IMPORTANCE
# --------------------------------------------------

print("\nFeature Importance:")

for name, importance in zip(
    features,
    model.feature_importances_
):

    print(
        f"{name:18s}: "
        f"{importance:.4f}"
    )


# --------------------------------------------------
# SAVE MODEL
# --------------------------------------------------

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

joblib.dump(
    model,
    MODEL_PATH
)

print(
    f"\nModel saved to: {MODEL_PATH}"
)