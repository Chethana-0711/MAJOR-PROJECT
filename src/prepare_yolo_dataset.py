from pathlib import Path
import shutil
import json

import cv2
import numpy as np
import yaml


# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_DIR / "MUDCAD-X" / "dataset"
OUTPUT_DIR = PROJECT_DIR / "data" / "yolo"

SPLIT_FILE = PROJECT_DIR / "data" / "capture_splits.json"


# --------------------------------------------------
# LOAD MASTER CAPTURE SPLIT
# --------------------------------------------------

if not SPLIT_FILE.exists():

    raise FileNotFoundError(
        f"Master split file not found:\n{SPLIT_FILE}\n\n"
        "Run src/capture_split.py first."
    )


with open(
    SPLIT_FILE,
    "r",
    encoding="utf-8"
) as f:

    split_data = json.load(f)


# --------------------------------------------------
# FIND CAPTURES FROM MASTER SPLIT
# --------------------------------------------------

splits = {}

for split_name in ["train", "val", "test"]:

    captures = []

    for relative_path in split_data[split_name]:

        capture_dir = (
            PROJECT_DIR / relative_path
        )

        vis_path = capture_dir / "vis.png"
        lwir_path = capture_dir / "lwir.png"
        label_path = capture_dir / "label.png"

        if not (
            vis_path.exists()
            and lwir_path.exists()
            and label_path.exists()
        ):

            raise FileNotFoundError(
                f"Incomplete capture:\n"
                f"{capture_dir}"
            )

        captures.append({
            "dir": capture_dir,
            "vis": vis_path,
            "lwir": lwir_path,
            "label": label_path
        })

    splits[split_name] = captures


# --------------------------------------------------
# PRINT SPLIT INFORMATION
# --------------------------------------------------

print("=" * 60)
print("USING MASTER CAPTURE SPLIT")
print("=" * 60)

print(
    f"Split file: {SPLIT_FILE}"
)

print(
    f"Seed: {split_data['seed']}"
)

print()

for split_name, items in splits.items():

    print(
        f"{split_name}: {len(items)} captures"
    )

print()


# --------------------------------------------------
# LOAD CLASS COLORS
# --------------------------------------------------

with open(
    DATASET_DIR / "labels_rgb.yaml",
    "r"
) as f:

    config = yaml.safe_load(f)


categories = config["categories"]


# Convert:
# class name -> class id

class_to_id = {
    name: idx
    for idx, name in enumerate(
        categories.keys()
    )
}


# Convert:
# RGB color -> class id

color_to_id = {
    tuple(color): class_to_id[name]
    for name, color in categories.items()
}


print("Classes:")

for name, class_id in class_to_id.items():

    print(
        f"{class_id}: {name}"
    )


# --------------------------------------------------
# CREATE OUTPUT DIRECTORIES
# --------------------------------------------------

for split_name in splits:

    (
        OUTPUT_DIR
        / "images"
        / split_name
    ).mkdir(
        parents=True,
        exist_ok=True
    )

    (
        OUTPUT_DIR
        / "lwir"
        / split_name
    ).mkdir(
        parents=True,
        exist_ok=True
    )

    (
        OUTPUT_DIR
        / "labels"
        / split_name
    ).mkdir(
        parents=True,
        exist_ok=True
    )


# --------------------------------------------------
# CONVERT MASK → YOLO BOXES
# --------------------------------------------------

def create_yolo_labels(
    label_path,
    output_label_path
):

    label_image = cv2.imread(
        str(label_path),
        cv2.IMREAD_COLOR
    )

    if label_image is None:

        raise ValueError(
            f"Could not read label image:\n"
            f"{label_path}"
        )


    # OpenCV loads images as BGR.
    # Convert to RGB because the YAML
    # class colors are RGB.

    label_image = cv2.cvtColor(
        label_image,
        cv2.COLOR_BGR2RGB
    )


    height, width = (
        label_image.shape[:2]
    )


    yolo_lines = []


    for rgb_color, class_id in (
        color_to_id.items()
    ):

        # Find pixels belonging to this class.

        mask = np.all(
            label_image == rgb_color,
            axis=2
        ).astype(np.uint8)


        if mask.sum() == 0:
            continue


        # Find disconnected objects separately.

        (
            num_labels,
            labels,
            stats,
            _
        ) = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )


        # Ignore component 0 because it is background.

        for component_id in range(
            1,
            num_labels
        ):

            x = stats[
                component_id,
                cv2.CC_STAT_LEFT
            ]

            y = stats[
                component_id,
                cv2.CC_STAT_TOP
            ]

            w = stats[
                component_id,
                cv2.CC_STAT_WIDTH
            ]

            h = stats[
                component_id,
                cv2.CC_STAT_HEIGHT
            ]

            area = stats[
                component_id,
                cv2.CC_STAT_AREA
            ]


            # Ignore tiny noise regions.

            if area < 20:
                continue


            # Convert bounding box
            # to YOLO format.

            x_center = (
                x + w / 2
            ) / width

            y_center = (
                y + h / 2
            ) / height

            box_width = (
                w / width
            )

            box_height = (
                h / height
            )


            yolo_lines.append(
                f"{class_id} "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{box_width:.6f} "
                f"{box_height:.6f}"
            )


    output_label_path.write_text(
        "\n".join(yolo_lines),
        encoding="utf-8"
    )


# --------------------------------------------------
# PROCESS DATASET
# --------------------------------------------------

for split_name, items in splits.items():

    print()
    print(
        f"Processing {split_name}..."
    )

    for index, capture in enumerate(items):

        # Create a unique filename.

        image_name = (
            f"{split_name}_{index:04d}.png"
        )


        vis_output = (
            OUTPUT_DIR
            / "images"
            / split_name
            / image_name
        )


        lwir_output = (
            OUTPUT_DIR
            / "lwir"
            / split_name
            / image_name
        )


        label_output = (
            OUTPUT_DIR
            / "labels"
            / split_name
            / f"{split_name}_{index:04d}.txt"
        )


        # Copy RGB image.

        shutil.copy2(
            capture["vis"],
            vis_output
        )


        # Copy LWIR image.

        shutil.copy2(
            capture["lwir"],
            lwir_output
        )


        # Convert segmentation mask
        # into YOLO bounding boxes.

        create_yolo_labels(
            capture["label"],
            label_output
        )


        if (
            index + 1
        ) % 50 == 0:

            print(
                f"  Processed "
                f"{index + 1}/"
                f"{len(items)}"
            )


# --------------------------------------------------
# VERIFY COUNTS
# --------------------------------------------------

print()
print("=" * 60)
print("YOLO DATASET PREPARATION COMPLETE")
print("=" * 60)

for split_name, items in splits.items():

    image_count = len(
        list(
            (
                OUTPUT_DIR
                / "images"
                / split_name
            ).glob("*.png")
        )
    )

    label_count = len(
        list(
            (
                OUTPUT_DIR
                / "labels"
                / split_name
            ).glob("*.txt")
        )
    )

    print(
        f"{split_name}: "
        f"{len(items)} captures | "
        f"{image_count} images | "
        f"{label_count} labels"
    )

print()
print(
    f"Output directory: {OUTPUT_DIR}"
)