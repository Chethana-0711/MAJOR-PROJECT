from pathlib import Path

from ultralytics import YOLO

from agents.spectral_agent import SpectralAgent
from agents.feature_agent import FeatureAgent
from agents.conflict_agent import ConflictAgent
from agents.uncertainty_agent import UncertaintyAgent
from agents.cognitive_agent import CognitiveAgent


# --------------------------------------------------
# PATHS
# --------------------------------------------------

MODEL_PATH = (
    "runs/detect/train/weights/best.pt"
)

DATASET_DIR = Path(
    "MUDCAD-X/dataset"
)


# --------------------------------------------------
# FIND A REAL CAPTURE
# --------------------------------------------------

rgb_path = next(
    DATASET_DIR.rglob("vis.png")
)

capture_dir = rgb_path.parent

lwir_path = (
    capture_dir /
    "lwir.png"
)


print("\nUsing capture:")
print(capture_dir)


# --------------------------------------------------
# LOAD AGENTS
# --------------------------------------------------

print("\nLoading agents...")

print("Loading Spatial Agent...")
spatial_model = YOLO(
    MODEL_PATH
)

print("Loading Spectral Agent...")
spectral_agent = SpectralAgent()

print("Loading Feature Agent...")
feature_agent = FeatureAgent()

print("Loading Conflict Agent...")
conflict_agent = ConflictAgent()

print("Loading Uncertainty Agent...")
uncertainty_agent = (
    UncertaintyAgent()
)

print("Loading Cognitive Agent...")
cognitive_agent = (
    CognitiveAgent()
)


# --------------------------------------------------
# SPATIAL AGENT
# --------------------------------------------------

print("\nRunning Spatial Agent...")

prediction = spatial_model.predict(
    source=str(rgb_path),
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

        "class":
            class_name,

        "bbox": [
            float(x1),
            float(y1),
            float(x2 - x1),
            float(y2 - y1)
        ],

        "confidence":
            confidence
    })


print(
    f"Spatial detections: "
    f"{len(spatial_predictions)}"
)


# --------------------------------------------------
# SPECTRAL AGENT
# --------------------------------------------------

print("\nRunning Spectral Agent...")

spectral_result = (
    spectral_agent.analyze(
        lwir_path
    )
)

spectral_regions = (
    spectral_result["regions"]
)

print(
    f"Spectral regions: "
    f"{len(spectral_regions)}"
)


# --------------------------------------------------
# FEATURE AGENT
# --------------------------------------------------

print("\nRunning Feature Agent...")

feature_result = (
    feature_agent.analyze(
        rgb_path=rgb_path,
        lwir_path=lwir_path
    )
)

feature_regions = (
    feature_result["regions"]
)

print(
    f"Feature regions: "
    f"{len(feature_regions)}"
)

print(
    f"Feature patches scanned: "
    f"{feature_result['num_patches']}"
)

print(
    f"Overall feature evidence: "
    f"{feature_result['overall_evidence']:.3f}"
)


# --------------------------------------------------
# CONFLICT AGENT
# --------------------------------------------------

print("\nRunning Conflict Agent...")

conflict_results = (
    conflict_agent.analyze(
        spatial_predictions,
        spectral_regions,
        feature_regions
    )
)

print(
    f"Candidates generated: "
    f"{len(conflict_results)}"
)


# --------------------------------------------------
# UNCERTAINTY AGENT
# --------------------------------------------------

print("\nRunning Uncertainty Agent...")

uncertainty_results = (
    uncertainty_agent.analyze(
        conflict_results
    )
)


# --------------------------------------------------
# COGNITIVE AGENT
# --------------------------------------------------

print("\nRunning Cognitive Agent...")

cognitive_results = (
    cognitive_agent.analyze(
        uncertainty_results
    )
)


# --------------------------------------------------
# FINAL RESULTS
# --------------------------------------------------

print("\n")
print("=" * 65)

print(
    "REAL MULTI-AGENT COGNITIVE ANALYSIS"
)

print("=" * 65)


if not cognitive_results:

    print(
        "\nNo candidates found."
    )

else:

    for i, result in enumerate(
        cognitive_results,
        start=1
    ):

        print(
            f"\nCandidate {i}"
        )

        print(
            "-" * 40
        )

        # ------------------------------------------
        # Identity
        # ------------------------------------------

        print(
            f"Class: "
            f"{result['class']}"
        )

        print(
            f"Bounding box: "
            f"{result['bbox']}"
        )

        # ------------------------------------------
        # Agent support
        # ------------------------------------------

        print(
            f"Agents supporting candidate: "
            f"{result['agent_count']}/3"
        )

        print(
            f"Spatial present: "
            f"{result['spatial_present']}"
        )

        print(
            f"Spectral present: "
            f"{result['spectral_present']}"
        )

        print(
            f"Feature present: "
            f"{result['feature_present']}"
        )

        # ------------------------------------------
        # Evidence
        # ------------------------------------------

        print(
            f"Spatial confidence: "
            f"{result['spatial_confidence']:.3f}"
        )

        print(
            f"Thermal evidence: "
            f"{result['thermal_evidence']:.3f}"
        )

        print(
            f"Feature evidence: "
            f"{result['feature_evidence']:.3f}"
        )

        print(
            f"Average evidence: "
            f"{result['average_evidence']:.3f}"
        )

        # ------------------------------------------
        # Conflict
        # ------------------------------------------

        print(
            f"Agreement: "
            f"{result['agreement']:.3f}"
        )

        print(
            f"Conflict: "
            f"{result['conflict']:.3f}"
        )

        # ------------------------------------------
        # Uncertainty
        # ------------------------------------------

        print(
            f"Uncertainty: "
            f"{result['uncertainty']:.3f}"
        )

        print(
            f"Uncertainty confidence: "
            f"{result['confidence']:.3f}"
        )

        print(
            f"Reliability: "
            f"{result['reliability']}"
        )

        # ------------------------------------------
        # Cognitive reasoning
        # ------------------------------------------

        print(
            f"\nCognitive decision: "
            f"{result['decision']}"
        )

        print(
            f"Cognitive confidence: "
            f"{result['confidence']:.3f}"
        )

        print(
            f"Meaningful conflict: "
            f"{result['meaningful_conflict']}"
        )

        print(
            f"Strongest agent: "
            f"{result['strongest_agent']}"
        )

        print(
            f"Strongest evidence: "
            f"{result['strongest_evidence']:.3f}"
        )

        print(
            "\nCognitive reasoning:"
        )

        print(
            result["interpretation"]
        )


print(
    "\nCognitive analysis complete."
)