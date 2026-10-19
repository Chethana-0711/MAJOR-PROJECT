from pathlib import Path

from ultralytics import YOLO


# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_DIR / "yolo26n.pt"
DATA_YAML = PROJECT_DIR / "data" / "yolo" / "data.yaml"

OUTPUT_DIR = PROJECT_DIR / "runs" / "detect"


# --------------------------------------------------
# TRAINING SETTINGS
# --------------------------------------------------

EPOCHS = 30
IMAGE_SIZE = 512
BATCH_SIZE = 4
DEVICE = 0
WORKERS = 2
SEED = 42


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    # --------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"YOLO model not found: {MODEL_PATH}"
        )

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Dataset configuration not found: {DATA_YAML}"
        )


    # --------------------------------------------------
    # LOAD PRETRAINED YOLO MODEL
    # --------------------------------------------------

    print("=" * 60)
    print("YOLO SPATIAL AGENT TRAINING")
    print("=" * 60)

    print("\nLoading pretrained model...")
    print(f"Model: {MODEL_PATH}")

    model = YOLO(str(MODEL_PATH))


    # --------------------------------------------------
    # TRAIN
    # --------------------------------------------------

    print("\nStarting training...")

    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=WORKERS,
        seed=SEED,
        pretrained=True,
        project=str(OUTPUT_DIR),
        name="spatial_train",
        exist_ok=False,
        plots=True,
    )


    # --------------------------------------------------
    # FINAL INFORMATION
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("YOLO TRAINING COMPLETE")
    print("=" * 60)

    best_model = (
        OUTPUT_DIR
        / "spatial_train"
        / "weights"
        / "best.pt"
    )

    print("\nBest model should be saved at:")
    print(best_model)

    print("\nTraining configuration:")
    print(f"Dataset:     {DATA_YAML}")
    print(f"Epochs:      {EPOCHS}")
    print(f"Image size:  {IMAGE_SIZE}")
    print(f"Batch size:  {BATCH_SIZE}")
    print(f"Device:      GPU {DEVICE}")
    print(f"Workers:     {WORKERS}")
    print(f"Seed:        {SEED}")

    print("\nTraining complete.")


# --------------------------------------------------
# WINDOWS ENTRY POINT


if __name__ == "__main__":
    main()