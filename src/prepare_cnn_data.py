from pathlib import Path
import random
import cv2
import numpy as np


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

DATASET_DIR = Path("MUDCAD-X/dataset")
OUTPUT_DIR = Path("data/cnn")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

PATCH_SIZE = 64
PATCHES_PER_CLASS = 20

SEED = 42


# --------------------------------------------------
# CREATE PATCH
# --------------------------------------------------

def extract_patch(image, x, y, size):
    half = size // 2

    h, w = image.shape[:2]

    x1 = max(0, x - half)
    y1 = max(0, y - half)
    x2 = min(w, x + half)
    y2 = min(h, y + half)

    patch = image[y1:y2, x1:x2]

    if patch.shape[0] != size or patch.shape[1] != size:
        patch = cv2.copyMakeBorder(
            patch,
            0,
            size - patch.shape[0],
            0,
            size - patch.shape[1],
            cv2.BORDER_REFLECT
        )

    return patch


# --------------------------------------------------
# FIND CAPTURES
# --------------------------------------------------

def find_captures():

    captures = []

    for label_path in DATASET_DIR.rglob("label.png"):

        capture = label_path.parent

        if (
            (capture / "vis.png").exists()
            and
            (capture / "lwir.png").exists()
        ):
            captures.append(capture)

    return sorted(captures)


# --------------------------------------------------
# SPLIT CAPTURES
# --------------------------------------------------

def split_captures(captures):

    random.seed(SEED)

    captures = captures.copy()

    random.shuffle(captures)

    total = len(captures)

    train_end = int(
        total * TRAIN_RATIO
    )

    val_end = train_end + int(
        total * VAL_RATIO
    )

    train = captures[:train_end]

    val = captures[
        train_end:val_end
    ]

    test = captures[val_end:]

    return train, val, test


# --------------------------------------------------
# PROCESS CAPTURE
# --------------------------------------------------

def process_capture(
    capture,
    output_dir,
    capture_id
):

    vis = cv2.imread(
        str(capture / "vis.png")
    )

    lwir = cv2.imread(
        str(capture / "lwir.png"),
        cv2.IMREAD_GRAYSCALE
    )

    label = cv2.imread(
        str(capture / "label.png")
    )

    if vis is None or lwir is None or label is None:
        return 0

    # Target = any non-black label pixel
    target_mask = np.any(
        label != 0,
        axis=2
    )

    background_mask = ~target_mask

    target_y, target_x = np.where(
        target_mask
    )

    background_y, background_x = np.where(
        background_mask
    )

    if len(target_x) == 0:
        return 0

    rng = np.random.default_rng(
        SEED + capture_id
    )

    target_count = min(
        PATCHES_PER_CLASS,
        len(target_x)
    )

    background_count = min(
        PATCHES_PER_CLASS,
        len(background_x)
    )

    target_indices = rng.choice(
        len(target_x),
        target_count,
        replace=False
    )

    background_indices = rng.choice(
        len(background_x),
        background_count,
        replace=False
    )

    count = 0

    # --------------------------------------------------
    # TARGET PATCHES
    # --------------------------------------------------

    for i in target_indices:

        x = int(target_x[i])
        y = int(target_y[i])

        vis_patch = extract_patch(
            vis,
            x,
            y,
            PATCH_SIZE
        )

        lwir_patch = extract_patch(
            lwir,
            x,
            y,
            PATCH_SIZE
        )

        filename = (
            f"{capture_id:04d}_"
            f"target_{count:03d}"
        )

        cv2.imwrite(
            str(
                output_dir /
                "target" /
                f"{filename}_rgb.jpg"
            ),
            vis_patch
        )

        cv2.imwrite(
            str(
                output_dir /
                "target" /
                f"{filename}_lwir.jpg"
            ),
            lwir_patch
        )

        count += 1

    # --------------------------------------------------
    # BACKGROUND PATCHES
    # --------------------------------------------------

    for i in background_indices:

        x = int(background_x[i])
        y = int(background_y[i])

        vis_patch = extract_patch(
            vis,
            x,
            y,
            PATCH_SIZE
        )

        lwir_patch = extract_patch(
            lwir,
            x,
            y,
            PATCH_SIZE
        )

        filename = (
            f"{capture_id:04d}_"
            f"background_{i:03d}"
        )

        cv2.imwrite(
            str(
                output_dir /
                "background" /
                f"{filename}_rgb.jpg"
            ),
            vis_patch
        )

        cv2.imwrite(
            str(
                output_dir /
                "background" /
                f"{filename}_lwir.jpg"
            ),
            lwir_patch
        )

        count += 1

    return count


# --------------------------------------------------
# PROCESS SPLIT
# --------------------------------------------------

def process_split(
    captures,
    split_name
):

    output_dir = (
        OUTPUT_DIR /
        split_name
    )

    target_dir = (
        output_dir /
        "target"
    )

    background_dir = (
        output_dir /
        "background"
    )

    target_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    background_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    total = 0

    for i, capture in enumerate(
        captures
    ):

        count = process_capture(
            capture,
            output_dir,
            i
        )

        total += count

        if (i + 1) % 100 == 0:

            print(
                f"  {split_name}: "
                f"{i + 1}/{len(captures)}"
            )

    print(
        f"  {split_name} patches: "
        f"{total}"
    )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print(
        "Preparing multimodal CNN dataset..."
    )

    captures = find_captures()

    print(
        f"Complete captures: "
        f"{len(captures)}"
    )

    train, val, test = split_captures(
        captures
    )

    print("\nCapture split:")

    print(
        f"train: {len(train)}"
    )

    print(
        f"val:   {len(val)}"
    )

    print(
        f"test:  {len(test)}"
    )

    process_split(
        train,
        "train"
    )

    process_split(
        val,
        "val"
    )

    process_split(
        test,
        "test"
    )

    print(
        "\nCNN dataset preparation complete!"
    )

    print(
        f"Output directory: "
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()