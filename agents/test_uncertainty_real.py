from pathlib import Path

from ultralytics import YOLO

from agents.spectral_agent import SpectralAgent
from agents.feature_agent import FeatureAgent
from agents.conflict_agent import ConflictAgent
from agents.uncertainty_agent import UncertaintyAgent


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = "runs/detect/train/weights/best.pt"

DATASET_DIR = Path("MUDCAD-X/dataset")


# ============================================================
# FIND A COMPLETE CAPTURE
# ============================================================

rgb_path = next(
    DATASET_DIR.rglob("vis.png")
)

capture_dir = rgb_path.parent

lwir_path = capture_dir / "lwir.png"


# ============================================================
# LOAD AGENTS
# ============================================================

print("\nLoading agents...")

print("Loading Spatial Agent...")
spatial_model = YOLO(MODEL_PATH)

print("Loading Spectral Agent...")
spectral_agent = SpectralAgent()

print("Loading Feature Agent...")
feature_agent = FeatureAgent()

print("Loading Conflict Agent...")
conflict_agent = ConflictAgent()

print("Loading Uncertainty Agent...")
uncertainty_agent = UncertaintyAgent()


# ============================================================
# SPATIAL AGENT
# ============================================================

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


print(
    f"Spatial detections: "
    f"{len(spatial_predictions)}"
)


# ============================================================
# SPECTRAL AGENT
# ============================================================

print("\nRunning Spectral Agent...")

spectral_result = spectral_agent.analyze(
    lwir_path
)

spectral_regions = spectral_result["regions"]


print(
    f"Spectral regions: "
    f"{len(spectral_regions)}"
)


# ============================================================
# FEATURE AGENT
# ============================================================

print("\nRunning Feature Agent...")

feature_result = feature_agent.analyze(
    rgb_path=rgb_path,
    lwir_path=lwir_path
)

feature_regions = feature_result["regions"]


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


# ============================================================
# CONFLICT AGENT
# ============================================================

print("\nRunning Conflict Agent...")

conflict_results = conflict_agent.analyze(
    spatial_predictions,
    spectral_regions,
    feature_regions
)

conflict_candidates = conflict_results["candidates"]


print(
    f"Candidates generated: "
    f"{len(conflict_candidates)}"
)


# ============================================================
# UNCERTAINTY AGENT
# ============================================================

print("\nRunning Uncertainty Agent...")

uncertainty_results = uncertainty_agent.analyze(
    conflict_results
)


# ============================================================
# RESULTS
# ============================================================

print("\n")
print("=" * 65)
print(
    "REAL MULTI-AGENT UNCERTAINTY ANALYSIS"
)
print("=" * 65)


if not uncertainty_results:

    print(
        "\nNo candidates found."
    )

else:

    for i, result in enumerate(
        uncertainty_results,
        start=1
    ):

        print(
            f"\nCandidate {i}"
        )

        print("-" * 40)

        print(
            f"Class: "
            f"{result['class']}"
        )

        print(
            f"Bounding box: "
            f"{result['bbox']}"
        )

        # ----------------------------------------------------
        # AGENT PARTICIPATION
        # ----------------------------------------------------

        print("\nAgent participation:")

        print(
            f"  Available agents: "
            f"{result['available_count']}/3"
        )

        print(
            f"  Supporting agents: "
            f"{result['support_count']}/"
            f"{result['available_count']}"
        )

        print(
            f"  Spatial: "
            f"{result['spatial_status']}"
        )

        print(
            f"  Thermal: "
            f"{result['spectral_status']}"
        )

        print(
            f"  Feature: "
            f"{result['feature_status']}"
        )

        # ----------------------------------------------------
        # EVIDENCE
        # ----------------------------------------------------

        print("\nEvidence:")

        print(
            f"  Spatial: "
            f"{result['spatial_evidence']:.3f}"
        )

        print(
            f"  Thermal: "
            f"{result['spectral_evidence']:.3f}"
        )

        print(
            f"  Feature: "
            f"{result['feature_evidence']:.3f}"
        )

        print(
            f"  Evidence strength: "
            f"{result['evidence_strength']:.3f}"
        )

        # ----------------------------------------------------
        # CONFLICT
        # ----------------------------------------------------

        print("\nConflict:")

        print(
            f"  Pattern: "
            f"{result['evidence_pattern']}"
        )

        print(
            f"  Agreement: "
            f"{result['agreement']:.3f}"
        )

        print(
            f"  Conflict: "
            f"{result['conflict']:.3f}"
        )

        print(
            f"  Conflict state: "
            f"{result['conflict_state']}"
        )

        # ----------------------------------------------------
        # UNCERTAINTY
        # ----------------------------------------------------

        print("\nUncertainty:")

        print(
            f"  Coverage uncertainty: "
            f"{result['coverage_uncertainty']:.3f}"
        )

        print(
            f"  Evidence uncertainty: "
            f"{result['evidence_uncertainty']:.3f}"
        )

        print(
            f"  Conflict uncertainty: "
            f"{result['conflict_uncertainty']:.3f}"
        )

        print(
            f"  Overall uncertainty: "
            f"{result['uncertainty']:.3f}"
        )

        print(
            f"  Confidence: "
            f"{result['confidence']:.3f}"
        )

        print(
            f"  Reliability: "
            f"{result['reliability']}"
        )


# ============================================================
# VISUALIZATION
# ============================================================

print("\n")
print("=" * 65)
print(
    "GENERATING UNCERTAINTY VISUALIZATION"
)
print("=" * 65)


uncertainty_agent.save_visualization(
    rgb_path,
    uncertainty_results,
    output_path="outputs/uncertainty_analysis.png"
)


print(
    "\nUncertainty analysis complete."
)