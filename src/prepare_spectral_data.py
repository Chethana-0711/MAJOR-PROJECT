from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

DATASET_DIR = Path("MUDCAD-X/dataset")
OUTPUT_DIR = Path("data/spectral")

MAX_SAMPLES_PER_CAPTURE = 200


# --------------------------------------------------
# EXTRACT FEATURES
# --------------------------------------------------

def extract_features(image, mask):
    """
    Extract thermal features from target/background
    pixels.

    Features:
    - normalized intensity
    - local mean
    - local standard deviation
    - local contrast
    """

    image = image.astype(np.float32) / 255.0

    local_mean = cv2.GaussianBlur(
        image,
        (15, 15),
        0
    )

    local_sq_mean = cv2.GaussianBlur(
        image ** 2,
        (15, 15),
        0
    )

    local_std = np.sqrt(
        np.maximum(
            local_sq_mean - local_mean ** 2,
            0
        )
    )

    local_contrast = np.abs(
        image - local_mean
    )

    pixels = np.where(mask)

    features = np.column_stack([
        image[pixels],
        local_mean[pixels],
        local_std[pixels],
        local_contrast[pixels]
    ])

    return features


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    rows = []

    captures = sorted(
        p.parent
        for p in DATASET_DIR.rglob("label.png")
        if (
            (p.parent / "lwir.png").exists()
            and
            (p.parent / "vis.png").exists()
        )
    )

    print(
        f"Complete captures: {len(captures)}"
    )

    rng = np.random.default_rng(42)

    for index, capture in enumerate(
        captures,
        start=1
    ):

        lwir_path = capture / "lwir.png"
        label_path = capture / "label.png"

        lwir = cv2.imread(
            str(lwir_path),
            cv2.IMREAD_GRAYSCALE
        )

        label = cv2.imread(
            str(label_path),
            cv2.IMREAD_COLOR
        )

        if lwir is None or label is None:
            continue

        # Any non-black pixel is target.
        target_mask = np.any(
            label != 0,
            axis=2
        )

        background_mask = ~target_mask

        target_features = extract_features(
            lwir,
            target_mask
        )

        background_features = extract_features(
            lwir,
            background_mask
        )

        # Limit samples so the background does not
        # overwhelm the target samples.
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

        for row in target_features[
            target_indices
        ]:

            rows.append([
                *row,
                1
            ])

        for row in background_features[
            background_indices
        ]:

            rows.append([
                *row,
                0
            ])

        if index % 100 == 0:

            print(
                f"Processed "
                f"{index}/{len(captures)}"
            )

    # --------------------------------------------------
    # SAVE
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

    output_path = (
        OUTPUT_DIR /
        "spectral_features.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    print("\nSpectral dataset complete.")
    print(
        f"Samples: {len(df)}"
    )

    print(
        f"Target samples: "
        f"{(df.target == 1).sum()}"
    )

    print(
        f"Background samples: "
        f"{(df.target == 0).sum()}"
    )

    print(
        f"Saved to: {output_path}"
    )


if __name__ == "__main__":
    main()