from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from agents.spectral_agent import SpectralAgent
from agents.fusion_agent import FusionAgent


MODEL_PATH = "runs/detect/train/weights/best.pt"
DATASET_DIR = Path("MUDCAD-X/dataset")
OUTPUT_PATH = Path("outputs/multi_agent_demo.jpg")


# --------------------------------------------------
# LOAD AGENTS
# --------------------------------------------------

print("Loading agents...")

spatial_model = YOLO(MODEL_PATH)
spectral_agent = SpectralAgent()
fusion_agent = FusionAgent()


# --------------------------------------------------
# SELECT A REAL CAPTURE
# --------------------------------------------------

label_path = next(DATASET_DIR.rglob("label.png"))
capture_dir = label_path.parent

vis_path = capture_dir / "vis.png"
lwir_path = capture_dir / "lwir.png"

print("Capture:", capture_dir)


# --------------------------------------------------
# SPATIAL AGENT
# --------------------------------------------------

print("Running Spatial Agent...")

prediction = spatial_model.predict(
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

    class_name = spatial_model.names[class_id]

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


# --------------------------------------------------
# SPECTRAL AGENT
# --------------------------------------------------

print("Running Spectral Agent...")

spectral_result = spectral_agent.analyze(
    lwir_path
)

spectral_regions = spectral_result["regions"]


# --------------------------------------------------
# FUSION AGENT
# --------------------------------------------------

print("Running Fusion Agent...")

fusion_results = fusion_agent.fuse(
    spatial_predictions,
    spectral_regions
)


# --------------------------------------------------
# LOAD IMAGES
# --------------------------------------------------

vis = cv2.imread(str(vis_path))
lwir = cv2.imread(
    str(lwir_path),
    cv2.IMREAD_GRAYSCALE
)

lwir = cv2.cvtColor(
    lwir,
    cv2.COLOR_GRAY2BGR
)


# --------------------------------------------------
# DRAW SPATIAL DETECTIONS
# --------------------------------------------------

for result in fusion_results:

    x, y, w, h = map(
        int,
        result["bbox"]
    )

    cv2.rectangle(
        vis,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

    label = (
        f"{result['class']} "
        f"{result['spatial_confidence']:.2f}"
    )

    cv2.putText(
        vis,
        label,
        (x, max(y - 5, 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 0),
        1
    )


# --------------------------------------------------
# DRAW THERMAL REGIONS
# --------------------------------------------------

# --------------------------------------------------
# DRAW ONLY THERMAL EVIDENCE
# ASSOCIATED WITH YOLO DETECTIONS
# --------------------------------------------------

for result in fusion_results:

    # Only show thermal region when
    # Fusion Agent found meaningful overlap
    if result["overlap"] <= 0:
        continue

    # Find the spectral region that produced
    # the best overlap with this YOLO box
    best_region = None
    best_iou = 0.0

    for region in spectral_regions:

        iou = fusion_agent.calculate_iou(
            result["bbox"],
            region["bbox"]
        )

        if iou > best_iou:
            best_iou = iou
            best_region = region

    if best_region is None:
        continue

    x, y, w, h = map(
        int,
        best_region["bbox"]
    )

    # Draw thermal evidence
    cv2.rectangle(
        lwir,
        (x, y),
        (x + w, y + h),
        (0, 0, 255),
        2
    )

    cv2.putText(
        lwir,
        f"Thermal {best_region['evidence']:.2f}",
        (x, max(y - 5, 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 0, 255),
        1
    )


# --------------------------------------------------
# RESIZE FOR DISPLAY
# --------------------------------------------------

vis_display = cv2.resize(
    vis,
    (512, 512)
)

lwir_display = cv2.resize(
    lwir,
    (512, 512)
)


# --------------------------------------------------
# CREATE RESULT PANEL
# --------------------------------------------------

panel_height = 220

panel = np.ones(
    (panel_height, 1024, 3),
    dtype=np.uint8
) * 255


cv2.putText(
    panel,
    "MULTI-AGENT CAMOUFLAGE DETECTION",
    (20, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.75,
    (0, 0, 0),
    2
)


cv2.putText(
    panel,
    f"Spatial detections: {len(spatial_predictions)}",
    (20, 65),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.5,
    (0, 0, 0),
    1
)


cv2.putText(
    panel,
    f"Thermal regions: {len(spectral_regions)}",
    (20, 90),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.5,
    (0, 0, 0),
    1
)


# Display fusion results
y_position = 120

for i, result in enumerate(
    fusion_results[:3],
    start=1
):

    text = (
        f"{i}. {result['class']} | "
        f"Spatial: {result['spatial_confidence']:.2f} | "
        f"Thermal: {result['thermal_evidence']:.2f} | "
        f"Conflict: {result['conflict']:.2f} | "
        f"Suspicion: {result['suspicion_score']:.2f} | "
        f"{result['decision']}"
    )

    cv2.putText(
        panel,
        text,
        (20, y_position),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0, 0, 0),
        1
    )

    y_position += 28


# --------------------------------------------------
# COMBINE EVERYTHING
# --------------------------------------------------

top_row = np.hstack(
    [vis_display, lwir_display]
)

final_image = np.vstack(
    [top_row, panel]
)


# --------------------------------------------------
# SAVE
# --------------------------------------------------

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

cv2.imwrite(
    str(OUTPUT_PATH),
    final_image
)


print("\nDemo complete!")
print("Saved:", OUTPUT_PATH)