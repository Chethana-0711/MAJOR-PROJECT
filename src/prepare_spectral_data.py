from pathlib import Path
import json

import cv2
import numpy as np
import pandas as pd


# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_DIR / "MUDCAD-X" / "dataset"
OUTPUT_DIR = PROJECT_DIR / "data" / "spectral"

SPLIT_FILE = PROJECT_DIR / "data" / "capture_splits.json"

MAX_SAMPLES_PER_CAPTURE = 200

SEED = 42


# --------------------------------------------------
# LOAD MASTER CAPTURE SPLIT
# --------------------------------------------------

if not SPLIT_FILE.exists():

    raise FileNotFoundError(
        f"Master split file not found:\n"
        f"{SPLIT_FILE}\n\n"
        f"Run src/capture_split.py first."
    )


with open(
    SPLIT_FILE,
    "r",
    encoding="utf-8"
) as f:

    split_data = json.load(f)


# --------------------------------------------------
# LOAD CAPTURES
# --------------------------------------------------

def load_split_captures(split_name):

    captures = []

    for relative_path in split_data[split_name]:

        capture = (
            PROJECT_DIR / relative_path
        )

        lwir_path = capture / "lwir.png"
        vis_path = capture / "vis.png"
        label_path = capture / "label.png"

        if not (
            lwir_path.exists()
            and vis_path.exists()
            and label_path.exists()
        ):

            raise FileNotFoundError(
                f"Incomplete capture:\n"
                f"{capture}"
            )

        captures.append(capture)

    return captures


splits = {

    "train": load_split_captures(
        "train"
    ),

    "val": load_split_captures(
        "val"
    ),

    "test": load_split_captures(
        "test"
    )
}


# --------------------------------------------------
# EXTRACT FEATURES
# --------------------------------------------------

def extract_features(
    image,
    mask
):

    """
    Extract thermal features from
    target/background pixels.

    Features:
    - normalized intensity
    - local mean
    - local standard deviation
    - local contrast
    """

    image = (
        image.astype(np.float32)
        / 255.0
    )


    # Local thermal background.

    local_mean = cv2.GaussianBlur(
        image,
        (15, 15),
        0
    )


    # Local squared mean.

    local_sq_mean = cv2.GaussianBlur(
        image ** 2,
        (15, 15),
        0
    )


    # Local standard deviation.

    local_std = np.sqrt(
        np.maximum(
            local_sq_mean
            - local_mean ** 2,
            0
        )
    )


    # Absolute local contrast.

    local_contrast = np.abs(
        image
        - local_mean
    )


    pixels = np.where(
        mask
    )


    features = np.column_stack([
        image[pixels],
        local_mean[pixels],
        local_std[pixels],
        local_contrast[pixels]
    ])


    return features


# --------------------------------------------------
# PROCESS ONE SPLIT
# --------------------------------------------------

def process_split(
    captures,
    split_name,
    seed_offset
):

    print()
    print(
        f"Processing {split_name}..."
    )


    rows = []


    rng = np.random.default_rng(
        SEED + seed_offset
    )


    valid_captures = 0


    for index, capture in enumerate(
        captures,
        start=1
    ):

        lwir_path = capture / "lwir.png"
        label_path = capture / "label.png"


        # --------------------------------------------------
        # LOAD IMAGES
        # --------------------------------------------------

        lwir = cv2.imread(
            str(lwir_path),
            cv2.IMREAD_GRAYSCALE
        )


        label = cv2.imread(
            str(label_path),
            cv2.IMREAD_COLOR
        )


        if (
            lwir is None
            or
            label is None
        ):

            print(
                f"  Skipping invalid capture: "
                f"{capture}"
            )

            continue


        # --------------------------------------------------
        # CREATE TARGET / BACKGROUND MASKS
        # --------------------------------------------------

        # Any non-black label pixel
        # is considered target.

        target_mask = np.any(
            label != 0,
            axis=2
        )


        background_mask = ~target_mask


        # --------------------------------------------------
        # EXTRACT FEATURES
        # --------------------------------------------------

        target_features = extract_features(
            lwir,
            target_mask
        )


        background_features = extract_features(
            lwir,
            background_mask
        )


        # --------------------------------------------------
        # SAMPLE FEATURES
        # --------------------------------------------------

        target_count = min(
            len(target_features),
            MAX_SAMPLES_PER_CAPTURE
        )


        background_count = min(
            len(background_features),
            MAX_SAMPLES_PER_CAPTURE
        )


        if target_count == 0:

            continue


        target_indices = rng.choice(
            len(target_features),
            size=target_count,
            replace=False
        )


        background_indices = rng.choice(
            len(background_features),
            size=background_count,
            replace=False
        )


        # --------------------------------------------------
        # TARGET SAMPLES
        # --------------------------------------------------

        for row in target_features[
            target_indices
        ]:

            rows.append([
                *row,
                1
            ])


        # --------------------------------------------------
        # BACKGROUND SAMPLES
        # --------------------------------------------------

        for row in background_features[
            background_indices
        ]:

            rows.append([
                *row,
                0
            ])


        valid_captures += 1


        if index % 100 == 0:

            print(
                f"  Processed "
                f"{index}/{len(captures)}"
            )


    # --------------------------------------------------
    # CREATE DATAFRAME
    # --------------------------------------------------

    df = pd.DataFrame(
        rows,
        columns=[
            "intensity",
            "local_mean",
            "local_std",
            "local_contrast",
            "target"
        ]
    )


    # --------------------------------------------------
    # SAVE SPLIT
    # --------------------------------------------------

    output_path = (
        OUTPUT_DIR
        / f"{split_name}_features.csv"
    )


    df.to_csv(
        output_path,
        index=False
    )


    # --------------------------------------------------
    # SUMMARY
    # --------------------------------------------------

    target_samples = int(
        (df.target == 1).sum()
    )


    background_samples = int(
        (df.target == 0).sum()
    )


    print()

    print(
        f"{split_name} complete."
    )

    print(
        f"  Captures: "
        f"{valid_captures}/{len(captures)}"
    )

    print(
        f"  Samples: "
        f"{len(df)}"
    )

    print(
        f"  Target samples: "
        f"{target_samples}"
    )

    print(
        f"  Background samples: "
        f"{background_samples}"
    )

    print(
        f"  Saved to: "
        f"{output_path}"
    )


    return df


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    print("=" * 60)
    print("USING MASTER CAPTURE SPLIT")
    print("=" * 60)


    print(
        f"Split file: "
        f"{SPLIT_FILE}"
    )


    print(
        f"Seed: "
        f"{split_data['seed']}"
    )


    print()


    for split_name, captures in (
        splits.items()
    ):

        print(
            f"{split_name}: "
            f"{len(captures)} captures"
        )


    print()


    # --------------------------------------------------
    # PROCESS TRAIN
    # --------------------------------------------------

    train_df = process_split(
        splits["train"],
        "train",
        seed_offset=0
    )


    # --------------------------------------------------
    # PROCESS VALIDATION
    # --------------------------------------------------

    val_df = process_split(
        splits["val"],
        "val",
        seed_offset=1
    )


    # --------------------------------------------------
    # PROCESS TEST
    # --------------------------------------------------

    test_df = process_split(
        splits["test"],
        "test",
        seed_offset=2
    )


    # --------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------

    print()
    print("=" * 60)
    print("SPECTRAL DATASET PREPARATION COMPLETE")
    print("=" * 60)


    print(
        f"Train: "
        f"{len(train_df)} samples"
    )


    print(
        f"Validation: "
        f"{len(val_df)} samples"
    )


    print(
        f"Test: "
        f"{len(test_df)} samples"
    )


    print()


    print(
        "Files created:"
    )


    print(
        f"  {OUTPUT_DIR / 'train_features.csv'}"
    )


    print(
        f"  {OUTPUT_DIR / 'val_features.csv'}"
    )


    print(
        f"  {OUTPUT_DIR / 'test_features.csv'}"
    )


if __name__ == "__main__":

    main()