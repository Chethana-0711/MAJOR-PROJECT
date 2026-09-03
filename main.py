from pathlib import Path

from ultralytics import YOLO

from agents.spectral_agent import SpectralAgent
from agents.fusion_agent import FusionAgent


# --------------------------------------------------
# PATHS
# --------------------------------------------------

MODEL_PATH = "runs/detect/train/weights/best.pt"

DATASET_DIR = Path("MUDCAD-X/dataset")


# --------------------------------------------------
# INITIALIZE AGENTS
# --------------------------------------------------

print("Loading Spatial Agent...")
spatial_model = YOLO(MODEL_PATH)

print("Loading Spectral Agent...")
spectral_agent = SpectralAgent()

print("Loading Fusion Agent...")
fusion_agent = FusionAgent()


# --------------------------------------------------
# FIND A REAL CAPTURE
# --------------------------------------------------

label_path = next(
    DATASET_DIR.rglob("label.png")
)

capture_dir = label_path.parent

vis_path = capture_dir / "vis.png"
lwir_path = capture_dir / "lwir.png"

print("\nUsing capture:")
print(capture_dir)


# --------------------------------------------------
# SPATIAL AGENT
# --------------------------------------------------

print("\nRunning Spatial Agent...")

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
        box.xyxy[0]
        .cpu()
        .numpy()
    )

    confidence = float(
        box.conf[0]
        .cpu()
        .numpy()
    )

    class_id = int(
        box.cls[0]
        .cpu()
        .numpy()
    )

    class_name = spatial_model.names[
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


print(
    f"Spatial detections: "
    f"{len(spatial_predictions)}"
)


# --------------------------------------------------
# SPECTRAL AGENT
# --------------------------------------------------

print("\nRunning Spectral Agent...")

spectral_result = spectral_agent.analyze(
    lwir_path
)

spectral_regions = spectral_result[
    "regions"
]

print(
    f"Thermal regions: "
    f"{len(spectral_regions)}"
)


# --------------------------------------------------
# FUSION AGENT
# --------------------------------------------------

print("\nRunning Fusion Agent...")

fusion_results = fusion_agent.fuse(
    spatial_predictions,
    spectral_regions
)


# --------------------------------------------------
# FINAL RESULTS
# --------------------------------------------------

print("\n")
print("=" * 50)
print("MULTI-AGENT CAMOUFLAGE DETECTION")
print("=" * 50)


if not fusion_results:

    print("\nNo spatial detections found.")

else:

    for i, result in enumerate(
        fusion_results,
        start=1
    ):

        print(
            f"\nDetection {i}"
        )

        print(
            f"Class: "
            f"{result['class']}"
        )

        print(
            f"Spatial confidence: "
            f"{result['spatial_confidence']:.3f}"
        )

        print(
            f"Thermal evidence: "
            f"{result['thermal_evidence']:.3f}"
        )

        print(
            f"Spatial overlap: "
            f"{result['overlap']:.3f}"
        )

        print(
            f"Agreement: "
            f"{result['agreement']:.3f}"
        )

        print(
            f"Conflict: "
            f"{result['conflict']:.3f}"
        )

        print(
            f"Uncertainty: "
            f"{result['uncertainty']:.3f}"
        )

        print(
            f"Suspicion score: "
            f"{result['suspicion_score']:.3f}"
        )

        print(
            f"Decision: "
            f"{result['decision']}"
        )


print("\nPipeline complete.")