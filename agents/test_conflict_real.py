from pathlib import Path

from agents.spatial_agent import SpatialAgent
from agents.spectral_agent import SpectralAgent
from agents.feature_agent import FeatureAgent
from agents.conflict_agent import ConflictAgent


# ============================================================
# PATHS
# ============================================================

DATASET_DIR = Path(
    "MUDCAD-X/dataset"
)


# ============================================================
# FIND ONE COMPLETE CAPTURE
# ============================================================

rgb_path = next(
    DATASET_DIR.rglob("vis.png")
)

capture_dir = rgb_path.parent

lwir_path = (
    capture_dir /
    "lwir.png"
)


# ============================================================
# LOAD AGENTS
# ============================================================

print("Loading agents...")

print("Loading Spatial Agent...")
spatial_agent = SpatialAgent()

print("Loading Spectral Agent...")
spectral_agent = SpectralAgent()

print("Loading Feature Agent...")
feature_agent = FeatureAgent()

print("Loading Conflict Agent...")
conflict_agent = ConflictAgent()


# ============================================================
# RUN SPATIAL
# ============================================================

print("\nRunning Spatial Agent...")

spatial_result = spatial_agent.analyze(
    rgb_path
)

# Spatial Agent currently returns detections
# either directly or inside a dictionary.

if isinstance(
    spatial_result,
    dict
):

    spatial_detections = spatial_result.get(
        "detections",
        []
    )

else:

    spatial_detections = spatial_result


print(
    f"Spatial detections: "
    f"{len(spatial_detections)}"
)


# ============================================================
# RUN SPECTRAL
# ============================================================

print("\nRunning Spectral Agent...")

spectral_result = spectral_agent.analyze(
    lwir_path
)

print(
    f"Spectral regions: "
    f"{len(spectral_result.get('regions', []))}"
)


# ============================================================
# RUN FEATURE
# ============================================================

print("\nRunning Feature Agent...")

feature_result = feature_agent.analyze(
    rgb_path=rgb_path,
    lwir_path=lwir_path
)

print(
    f"Feature regions: "
    f"{len(feature_result.get('regions', []))}"
)

print(
    f"Feature patches scanned: "
    f"{feature_result.get('num_patches', 0)}"
)

print(
    f"Overall feature evidence: "
    f"{feature_result.get('overall_evidence', 0.0):.3f}"
)


# ============================================================
# RUN CONFLICT
# ============================================================

print("\nRunning Conflict Agent...")

conflict_results = conflict_agent.analyze(
    spatial_result=spatial_result,
    spectral_result=spectral_result,
    feature_result=feature_result
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 65)
print("REAL MULTI-AGENT CONFLICT ANALYSIS")
print("=" * 65)

print(
    f"\nAvailable agents: "
    f"{conflict_results['available_agent_count']}/3"
)

print(
    "Agents: "
    +
    ", ".join(
        conflict_results["available_agents"]
    )
)

print(
    f"Candidates analysed: "
    f"{conflict_results['candidate_count']}"
)


# IMPORTANT:
#
# conflict_results is a dictionary.
# The actual candidates are inside:
#
# conflict_results["candidates"]

candidates = conflict_results[
    "candidates"
]


for index, result in enumerate(
    candidates,
    start=1
):

    print(
        "\n"
        +
        "=" * 40
    )

    print(
        f"Candidate {index}"
    )

    print(
        "-" * 40
    )

    print(
        f"Class: "
        f"{result['class']}"
    )

    print(
        f"Bounding box: "
        f"{result['bbox']}"
    )

    print(
        "\nAgent Status:"
    )

    print(
        f"  Spatial:  "
        f"{result['spatial_status']}"
    )

    print(
        f"  Thermal:  "
        f"{result['spectral_status']}"
    )

    print(
        f"  Feature:  "
        f"{result['feature_status']}"
    )

    print(
        "\nEvidence:"
    )

    print(
        f"  Spatial:  "
        f"{result['spatial_evidence']:.3f}"
    )

    print(
        f"  Thermal:  "
        f"{result['spectral_evidence']:.3f}"
    )

    print(
        f"  Feature:  "
        f"{result['feature_evidence']:.3f}"
    )

    print(
        "\nAgent participation:"
    )

    print(
        f"  Available: "
        f"{result['available_count']}/3"
    )

    print(
        f"  Supporting: "
        f"{result['support_count']}/"
        f"{result['available_count']}"
    )

    if result["missing_agents"]:

        print(
            "  Not available: "
            +
            ", ".join(
                result["missing_agents"]
            )
        )

    if result["non_detecting_agents"]:

        print(
            "  No detection: "
            +
            ", ".join(
                result["non_detecting_agents"]
            )
        )

    print(
        "\nConflict Analysis:"
    )

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

    print(
        f"  Average evidence: "
        f"{result['average_evidence']:.3f}"
    )


# ============================================================
# SAVE VISUALIZATION
# ============================================================

print("\n")
print("=" * 65)
print("GENERATING CONFLICT VISUALIZATION")
print("=" * 65)

visualization_path = (
    conflict_agent.save_visualization(
        rgb_path=rgb_path,
        results=conflict_results,
        output_path="outputs/conflict_analysis.png"
    )
)

print(
    f"\nConflict visualization saved to:"
    f"\n{visualization_path}"
)


# ============================================================
# DONE
# ============================================================

print("\n")
print("=" * 65)
print("CONFLICT AGENT TEST COMPLETE")
print("=" * 65)