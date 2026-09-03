from pathlib import Path

import cv2
import numpy as np

from feature_agent import FeatureAgent


# --------------------------------------------------
# PATHS
# --------------------------------------------------

DATASET = Path("MUDCAD-X/dataset")

OUTPUT = Path(
    "outputs/feature_detection.png"
)


# --------------------------------------------------
# LOAD SAMPLE
# --------------------------------------------------

rgb_path = next(
    DATASET.rglob("vis.png")
)

lwir_path = (
    rgb_path.parent /
    "lwir.png"
)


# --------------------------------------------------
# RUN FEATURE AGENT
# --------------------------------------------------

agent = FeatureAgent()

result = agent.analyze(
    rgb_path=rgb_path,
    lwir_path=lwir_path
)


# --------------------------------------------------
# LOAD RGB FOR DISPLAY
# --------------------------------------------------

image = cv2.imread(
    str(rgb_path)
)

if image is None:

    raise ValueError(
        "Could not load RGB image."
    )


# --------------------------------------------------
# DRAW FEATURE REGIONS
# --------------------------------------------------

for i, region in enumerate(
    result["regions"][:20],
    start=1
):

    x, y, w, h = region["bbox"]

    evidence = region["evidence"]

    cv2.rectangle(
        image,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

    label = (
        f"Feature {i}: "
        f"{evidence:.2f}"
    )

    cv2.putText(
        image,
        label,
        (x, max(y - 8, 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
        cv2.LINE_AA
    )


# --------------------------------------------------
# ADD SUMMARY
# --------------------------------------------------

summary = (
    f"Feature Agent | "
    f"Regions: {len(result['regions'])} | "
    f"Evidence: {result['overall_evidence']:.3f}"
)

cv2.putText(
    image,
    summary,
    (10, 25),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.65,
    (255, 255, 255),
    2,
    cv2.LINE_AA
)


# --------------------------------------------------
# SAVE
# --------------------------------------------------

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

cv2.imwrite(
    str(OUTPUT),
    image
)

print(
    f"Saved visualization to: {OUTPUT}"
)