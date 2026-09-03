from pathlib import Path
import random
import shutil

import cv2
import numpy as np
import yaml


# --------------------------------------------------
# PATHS
# --------------------------------------------------

DATASET_DIR = Path("MUDCAD-X/dataset")
OUTPUT_DIR = Path("data/yolo")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42


# --------------------------------------------------
# LOAD CLASS COLORS
# --------------------------------------------------

with open(DATASET_DIR / "labels_rgb.yaml", "r") as f:
    config = yaml.safe_load(f)

categories = config["categories"]

# Convert:
# class name -> class id
class_to_id = {
    name: idx for idx, name in enumerate(categories.keys())
}

# Convert:
# RGB color -> class id
color_to_id = {
    tuple(color): class_to_id[name]
    for name, color in categories.items()
}


print("Classes:")
for name, class_id in class_to_id.items():
    print(f"{class_id}: {name}")


# --------------------------------------------------
# FIND COMPLETE CAPTURES
# --------------------------------------------------

captures = []

for label_path in DATASET_DIR.rglob("label.png"):

    capture_dir = label_path.parent

    vis_path = capture_dir / "vis.png"
    lwir_path = capture_dir / "lwir.png"

    if vis_path.exists() and lwir_path.exists():

        captures.append({
            "dir": capture_dir,
            "vis": vis_path,
            "lwir": lwir_path,
            "label": label_path
        })


print(f"\nComplete captures found: {len(captures)}")


# --------------------------------------------------
# TRAIN / VAL / TEST SPLIT
# --------------------------------------------------

random.seed(SEED)
random.shuffle(captures)

total = len(captures)

train_end = int(total * TRAIN_RATIO)
val_end = train_end + int(total * VAL_RATIO)

splits = {
    "train": captures[:train_end],
    "val": captures[train_end:val_end],
    "test": captures[val_end:]
}


print("\nDataset split:")
for split, items in splits.items():
    print(f"{split}: {len(items)}")


# --------------------------------------------------
# CREATE OUTPUT DIRECTORIES
# --------------------------------------------------

for split in splits:

    (OUTPUT_DIR / "images" / split).mkdir(
        parents=True,
        exist_ok=True
    )

    (OUTPUT_DIR / "lwir" / split).mkdir(
        parents=True,
        exist_ok=True
    )

    (OUTPUT_DIR / "labels" / split).mkdir(
        parents=True,
        exist_ok=True
    )


# --------------------------------------------------
# CONVERT MASK → YOLO BOXES
# --------------------------------------------------

def create_yolo_labels(label_path, output_label_path):

    label_image = cv2.imread(
        str(label_path),
        cv2.IMREAD_COLOR
    )

    # OpenCV loads images as BGR.
    # Convert to RGB because YAML colors are RGB.
    label_image = cv2.cvtColor(
        label_image,
        cv2.COLOR_BGR2RGB
    )

    height, width = label_image.shape[:2]

    yolo_lines = []

    for rgb_color, class_id in color_to_id.items():

        # Find pixels belonging to this class.
        mask = np.all(
            label_image == rgb_color,
            axis=2
        ).astype(np.uint8)

        if mask.sum() == 0:
            continue

        # Find disconnected objects separately.
        num_labels, labels, stats, _ = (
            cv2.connectedComponentsWithStats(
                mask,
                connectivity=8
            )
        )

        # Ignore component 0 because it is background.
        for component_id in range(1, num_labels):

            x = stats[component_id, cv2.CC_STAT_LEFT]
            y = stats[component_id, cv2.CC_STAT_TOP]

            w = stats[component_id, cv2.CC_STAT_WIDTH]
            h = stats[component_id, cv2.CC_STAT_HEIGHT]

            area = stats[
                component_id,
                cv2.CC_STAT_AREA
            ]

            # Ignore tiny noise regions.
            if area < 20:
                continue

            # Convert bounding box to YOLO format.
            x_center = (x + w / 2) / width
            y_center = (y + h / 2) / height

            box_width = w / width
            box_height = h / height

            yolo_lines.append(
                f"{class_id} "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{box_width:.6f} "
                f"{box_height:.6f}"
            )

    output_label_path.write_text(
        "\n".join(yolo_lines)
    )


# --------------------------------------------------
# PROCESS DATASET
# --------------------------------------------------

for split, items in splits.items():

    print(f"\nProcessing {split}...")

    for index, capture in enumerate(items):

        # Create a unique filename.
        image_name = f"{split}_{index:04d}.png"

        vis_output = (
            OUTPUT_DIR
            / "images"
            / split
            / image_name
        )

        lwir_output = (
            OUTPUT_DIR
            / "lwir"
            / split
            / image_name
        )

        label_output = (
            OUTPUT_DIR
            / "labels"
            / split
            / f"{split}_{index:04d}.txt"
        )

        shutil.copy2(
            capture["vis"],
            vis_output
        )

        shutil.copy2(
            capture["lwir"],
            lwir_output
        )

        create_yolo_labels(
            capture["label"],
            label_output
        )

        if (index + 1) % 50 == 0:
            print(
                f"  Processed {index + 1}/{len(items)}"
            )


print("\nDataset preparation complete!")
print(f"Output directory: {OUTPUT_DIR}")