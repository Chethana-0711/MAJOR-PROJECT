from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data" / "spectral"

TRAIN_PATH = DATA_DIR / "train_features.csv"
VAL_PATH = DATA_DIR / "val_features.csv"
TEST_PATH = DATA_DIR / "test_features.csv"

MODEL_DIR = PROJECT_DIR / "models"
MODEL_PATH = MODEL_DIR / "spectral_rf.pkl"


# --------------------------------------------------
# FEATURES
# --------------------------------------------------

features = [
    "intensity",
    "local_mean",
    "local_std",
    "local_contrast",
]


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("=" * 60)
print("SPECTRAL RANDOM FOREST TRAINING")
print("=" * 60)

print("\nLoading datasets...")


train_df = pd.read_csv(
    TRAIN_PATH
)

val_df = pd.read_csv(
    VAL_PATH
)

test_df = pd.read_csv(
    TEST_PATH
)


print(
    f"Training samples:   {len(train_df)}"
)

print(
    f"Validation samples: {len(val_df)}"
)

print(
    f"Test samples:       {len(test_df)}"
)


# --------------------------------------------------
# FEATURES / LABELS
# --------------------------------------------------

X_train = train_df[
    features
]

y_train = train_df[
    "target"
]


X_val = val_df[
    features
]

y_val = val_df[
    "target"
]


X_test = test_df[
    features
]

y_test = test_df[
    "target"
]


# --------------------------------------------------
# RANDOM FOREST


print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced",
)


model.fit(
    X_train,
    y_train
)


print(
    "Training complete."
)


# --------------------------------------------------
# VALIDATION
# --------------------------------------------------

print("\n")
print("=" * 60)
print("VALIDATION RESULTS")
print("=" * 60)


val_predictions = model.predict(
    X_val
)


val_accuracy = accuracy_score(
    y_val,
    val_predictions
)


print(
    f"\nValidation Accuracy: "
    f"{val_accuracy:.4f}"
)


print("\nValidation Classification Report:")

print(
    classification_report(
        y_val,
        val_predictions,
        target_names=[
            "Background",
            "Target"
        ]
    )
)


print("Validation Confusion Matrix:")

print(
    confusion_matrix(
        y_val,
        val_predictions
    )
)


# --------------------------------------------------
# FINAL TEST


print("\n")
print("=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)


test_predictions = model.predict(
    X_test
)


test_accuracy = accuracy_score(
    y_test,
    test_predictions
)


print(
    f"\nTest Accuracy: "
    f"{test_accuracy:.4f}"
)


print("\nTest Classification Report:")

print(
    classification_report(
        y_test,
        test_predictions,
        target_names=[
            "Background",
            "Target"
        ]
    )
)


print("Test Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        test_predictions
    )
)


# --------------------------------------------------
# FEATURE IMPORTANCE
# --------------------------------------------------

print("\n")
print("=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)


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


print("\n")
print(
    f"Model saved to: "
    f"{MODEL_PATH}"
)


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

print("\n")
print("=" * 60)
print("TRAINING SUMMARY")
print("=" * 60)

print(
    f"Training captures:     597"
)

print(
    f"Validation captures:   127"
)

print(
    f"Test captures:         129"
)

print()

print(
    f"Validation Accuracy: "
    f"{val_accuracy:.4f}"
)

print(
    f"Test Accuracy:       "
    f"{test_accuracy:.4f}"
)

print()
print(
    "Random Forest training complete."
)