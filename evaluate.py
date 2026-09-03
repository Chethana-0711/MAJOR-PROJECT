from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from agents.spectral_agent import SpectralAgent
from agents.fusion_agent import FusionAgent


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

MODEL_PATH = "runs/detect/train/weights/best.pt"
TEST_DIR = Path("data/yolo/images/test")

DATASET_DIR = Path("MUDCAD-X/dataset")

IOU_THRESHOLD = 0.50


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def calculate_iou(box1, box2):
    """
    Boxes are [x, y, width, height].
    """

    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xa = max(x1, x2)
    ya = max(y1, y2)

    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)

    intersection = max(0, xb - xa) * max(0, yb - ya)

    area1 = w1 * h1
    area2 = w2 * h2

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def mask_to_boxes(label_path):
    """
    Convert colored ground-truth mask
    into bounding boxes.

    Each non-black connected component
    becomes one ground-truth object.
    """

    label = cv2.imread(
        str(label_path),
        cv2.IMREAD_COLOR
    )

    if label is None:
        return []

    # Any non-black pixel belongs to a target.
    mask = np.any(label != 0, axis=2).astype(np.uint8)

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )

    boxes = []

    for i in range(1, num_labels):

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        area = stats[i, cv2.CC_STAT_AREA]

        if area < 20:
            continue

        boxes.append([
            float(x),
            float(y),
            float(w),
            float(h)
        ])

    return boxes


# --------------------------------------------------
# LOAD AGENTS
# --------------------------------------------------

print("Loading model and agents...")

model = YOLO(MODEL_PATH)
spectral_agent = SpectralAgent()
fusion_agent = FusionAgent()


# --------------------------------------------------
# FIND TEST IMAGES
# --------------------------------------------------

test_images = sorted(
    TEST_DIR.glob("*.png")
)

print(
    f"Test images found: {len(test_images)}"
)

if not test_images:
    raise RuntimeError(
        "No test images found."
    )


# --------------------------------------------------
# METRICS
# --------------------------------------------------

total_gt = 0
total_predictions = 0
total_tp = 0

total_fusion = 0

high_suspicion = 0
moderate_suspicion = 0
low_suspicion = 0

spatial_confidences = []
thermal_evidences = []
conflicts = []
uncertainties = []
suspicion_scores = []


# --------------------------------------------------
# PROCESS TEST SET
# --------------------------------------------------

for index, image_path in enumerate(
    test_images,
    start=1
):

    # ----------------------------------------------
    # Find original capture
    # ----------------------------------------------

    image_name = image_path.stem

    # Our preparation script creates names such as
    # test_0000.png.
    # The original capture mapping is stored in the
    # preparation metadata if available.
    #
    # Instead, search for the corresponding VIS image.
    # ----------------------------------------------

    vis_matches = list(
        DATASET_DIR.rglob("vis.png")
    )

    # Match using image ordering created by
    # prepare_yolo_dataset.py.
    #
    # We recreate the sorted complete captures.
    complete_captures = []

    for label_path in DATASET_DIR.rglob(
        "label.png"
    ):

        capture = label_path.parent

        if (
            (capture / "vis.png").exists()
            and
            (capture / "lwir.png").exists()
        ):
            complete_captures.append(capture)

    complete_captures = sorted(
        complete_captures,
        key=lambda p: str(p)
    )

    # Determine split index.
    # Test contains the final 129 captures
    # according to our preparation split.
    test_start = 597 + 127

    capture_index = test_start + (index - 1)

    if capture_index >= len(complete_captures):
        continue

    capture_dir = complete_captures[
        capture_index
    ]

    vis_path = capture_dir / "vis.png"
    lwir_path = capture_dir / "lwir.png"
    label_path = capture_dir / "label.png"


    # ----------------------------------------------
    # Ground truth
    # ----------------------------------------------

    gt_boxes = mask_to_boxes(label_path)

    total_gt += len(gt_boxes)


    # ----------------------------------------------
    # Spatial Agent
    # ----------------------------------------------

    prediction = model.predict(
        source=str(vis_path),
        imgsz=512,
        conf=0.25,
        device=0,
        verbose=False
    )[0]

    spatial_predictions = []

    for box in prediction.boxes:

        x1, y1, x2, y2 = (
            box.xyxy[0].cpu().numpy()
        )

        confidence = float(
            box.conf[0].cpu().numpy()
        )

        class_id = int(
            box.cls[0].cpu().numpy()
        )

        class_name = model.names[
            class_id
        ]

        spatial_predictions.append({
            "class": class_name,
            "bbox": [
                float(x1),
                float(y1),
                float(x2 - x1),
                float(y2 - y1)
            ],
            "confidence": confidence
        })


    total_predictions += len(
        spatial_predictions
    )


    # ----------------------------------------------
    # Match YOLO predictions to GT
    # ----------------------------------------------

    matched_gt = set()

    for prediction_item in spatial_predictions:

        best_iou = 0.0
        best_gt = -1

        for gt_index, gt_box in enumerate(
            gt_boxes
        ):

            if gt_index in matched_gt:
                continue

            iou = calculate_iou(
                prediction_item["bbox"],
                gt_box
            )

            if iou > best_iou:
                best_iou = iou
                best_gt = gt_index

        if best_iou >= IOU_THRESHOLD:

            total_tp += 1
            matched_gt.add(best_gt)


    # ----------------------------------------------
    # Spectral Agent
    # ----------------------------------------------

    spectral_result = spectral_agent.analyze(
        lwir_path
    )

    spectral_regions = spectral_result[
        "regions"
    ]


    # ----------------------------------------------
    # Fusion Agent
    # ----------------------------------------------

    fusion_results = fusion_agent.fuse(
        spatial_predictions,
        spectral_regions
    )

    total_fusion += len(
        fusion_results
    )


    # ----------------------------------------------
    # Collect fusion statistics
    # ----------------------------------------------

    for result in fusion_results:

        spatial_confidences.append(
            result["spatial_confidence"]
        )

        thermal_evidences.append(
            result["thermal_evidence"]
        )

        conflicts.append(
            result["conflict"]
        )

        uncertainties.append(
            result["uncertainty"]
        )

        suspicion_scores.append(
            result["suspicion_score"]
        )

        if result["decision"] == "HIGH SUSPICION":
            high_suspicion += 1

        elif result["decision"] == "MODERATE SUSPICION":
            moderate_suspicion += 1

        else:
            low_suspicion += 1


    if index % 20 == 0:

        print(
            f"Processed {index}/"
            f"{len(test_images)}"
        )


# --------------------------------------------------
# FINAL METRICS
# --------------------------------------------------

false_positives = (
    total_predictions - total_tp
)

false_negatives = (
    total_gt - total_tp
)

precision = (
    total_tp /
    total_predictions
    if total_predictions > 0
    else 0
)

recall = (
    total_tp /
    total_gt
    if total_gt > 0
    else 0
)

f1 = (
    2 * precision * recall /
    (precision + recall)
    if precision + recall > 0
    else 0
)


# --------------------------------------------------
# REPORT
# --------------------------------------------------

print("\n")
print("=" * 60)
print("MULTI-AGENT TEST SET EVALUATION")
print("=" * 60)

print(
    f"\nTest images:       {len(test_images)}"
)

print(
    f"Ground-truth objects: {total_gt}"
)

print(
    f"YOLO predictions:      {total_predictions}"
)

print(
    f"True positives:        {total_tp}"
)

print(
    f"False positives:       {false_positives}"
)

print(
    f"False negatives:       {false_negatives}"
)

print("\nSPATIAL AGENT METRICS")
print("----------------------------")

print(
    f"Precision: {precision:.3f}"
)

print(
    f"Recall:    {recall:.3f}"
)

print(
    f"F1 Score:  {f1:.3f}"
)


print("\nFUSION RESULTS")
print("----------------------------")

print(
    f"Fusion detections: {total_fusion}"
)

print(
    f"High suspicion:    {high_suspicion}"
)

print(
    f"Moderate suspicion:{moderate_suspicion}"
)

print(
    f"Low suspicion:     {low_suspicion}"
)


if suspicion_scores:

    print("\nFUSION STATISTICS")
    print("----------------------------")

    print(
        f"Average spatial confidence: "
        f"{np.mean(spatial_confidences):.3f}"
    )

    print(
        f"Average thermal evidence:   "
        f"{np.mean(thermal_evidences):.3f}"
    )

    print(
        f"Average conflict:            "
        f"{np.mean(conflicts):.3f}"
    )

    print(
        f"Average uncertainty:         "
        f"{np.mean(uncertainties):.3f}"
    )

    print(
        f"Average suspicion score:     "
        f"{np.mean(suspicion_scores):.3f}"
    )

print("\nEvaluation complete.")