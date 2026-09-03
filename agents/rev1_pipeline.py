from pathlib import Path
import argparse
import shutil

import cv2
import numpy as np

from agents.spatial_agent import SpatialAgent
from agents.spectral_agent import SpectralAgent
from agents.feature_agent import FeatureAgent
from agents.conflict_agent import ConflictAgent
from agents.uncertainty_agent import UncertaintyAgent


# ============================================================
# REVIEW 1 MULTI-AGENT PIPELINE
# ============================================================

PROJECT_DIR = Path(".")
DATASET_DIR = PROJECT_DIR / "MUDCAD-X" / "dataset"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "review1"
MODEL_PATH = (
    PROJECT_DIR / "runs" / "detect" / "train" / "weights" / "best.pt"
)


# ============================================================
# FIND COMPLETE RGB + THERMAL CAPTURES
# ============================================================

def find_captures():

    captures = []

    for rgb_path in DATASET_DIR.rglob("vis.png"):

        capture_dir = rgb_path.parent
        lwir_path = capture_dir / "lwir.png"

        if lwir_path.exists():

            captures.append({
                "rgb": rgb_path,
                "lwir": lwir_path,
                "directory": capture_dir
            })

    captures.sort(
        key=lambda item: str(item["directory"]).lower()
    )

    return captures


# ============================================================
# SAVE CAPTURE INDEX
# ============================================================

def save_capture_index(captures):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    index_path = OUTPUT_DIR / "capture_index.txt"

    with open(
        index_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write("REVIEW 1 CAPTURE INDEX\n")
        file.write("=" * 80 + "\n\n")

        for index, capture in enumerate(
            captures,
            start=1
        ):

            file.write(
                f"{index:03d} | "
                f"{capture['directory']}\n"
            )

    return index_path


# ============================================================
# SELECT CAPTURE
# ============================================================

def select_capture(captures, requested_index=None):

    if requested_index is not None:

        if not 1 <= requested_index <= len(captures):

            raise ValueError(
                f"Capture number must be between "
                f"1 and {len(captures)}."
            )

        return (
            captures[requested_index - 1],
            requested_index
        )

    print("\n")
    print("=" * 70)
    print("REVIEW 1 - CAPTURE SELECTION")
    print("=" * 70)

    print(
        f"\nComplete RGB + Thermal captures found: "
        f"{len(captures)}"
    )

    print(
        "\nCapture list saved to:"
    )

    print(
        f"  {OUTPUT_DIR / 'capture_index.txt'}"
    )

    print(
        "\nEnter a capture number."
    )

    print(
        "Press ENTER to use capture #1."
    )

    while True:

        user_input = input(
            "\nEnter capture number: "
        ).strip()

        if user_input == "":
            index = 1

        else:

            try:
                index = int(user_input)

            except ValueError:

                print(
                    "Please enter a valid number."
                )
                continue

        if 1 <= index <= len(captures):

            return (
                captures[index - 1],
                index
            )

        print(
            f"Please enter a number from "
            f"1 to {len(captures)}."
        )


# ============================================================
# SAVE SELECTED INPUTS
# ============================================================

def save_inputs(rgb_path, lwir_path):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    shutil.copy2(
        rgb_path,
        OUTPUT_DIR / "01_input_rgb.png"
    )

    shutil.copy2(
        lwir_path,
        OUTPUT_DIR / "02_input_lwir.png"
    )


# ============================================================
# CREATE REVIEW 1 SUMMARY
# ============================================================

def create_summary(
    selected_index,
    capture_dir,
    spatial_result,
    spectral_result,
    feature_result,
    conflict_result,
    uncertainty_result
):

    spatial_count = len(
        spatial_result.get(
            "detections",
            []
        )
    )

    spectral_count = len(
        spectral_result.get(
            "regions",
            []
        )
    )

    feature_count = len(
        feature_result.get(
            "regions",
            []
        )
    )

    conflict_candidates = conflict_result.get(
        "candidates",
        []
    )

    candidate_count = len(
        conflict_candidates
    )

    uncertainty_values = [
        candidate.get(
            "uncertainty",
            0.0
        )
        for candidate in uncertainty_result
    ]

    confidence_values = [
        candidate.get(
            "confidence",
            0.0
        )
        for candidate in uncertainty_result
    ]

    average_uncertainty = (
        float(np.mean(uncertainty_values))
        if uncertainty_values
        else 0.0
    )

    average_confidence = (
        float(np.mean(confidence_values))
        if confidence_values
        else 0.0
    )

    conflict_states = {
        "LOW": 0,
        "MODERATE": 0,
        "HIGH": 0,
        "UNKNOWN": 0
    }

    for candidate in conflict_candidates:

        state = candidate.get(
            "conflict_state",
            "UNKNOWN"
        )

        if state in conflict_states:
            conflict_states[state] += 1

    reliability_counts = {
        "HIGH": 0,
        "MEDIUM": 0,
        "LOW": 0
    }

    for candidate in uncertainty_result:

        reliability = candidate.get(
            "reliability",
            "LOW"
        )

        if reliability in reliability_counts:
            reliability_counts[reliability] += 1

    width = 1200
    height = 520

    summary = np.zeros(
        (height, width, 3),
        dtype=np.uint8
    )

    # Header
    cv2.putText(
        summary,
        "REVIEW 1 - MULTI-AGENT CAMOUFLAGE ANALYSIS",
        (35, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        summary,
        f"Capture #{selected_index}  |  {capture_dir}",
        (35, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (190, 190, 190),
        1,
        cv2.LINE_AA
    )

    # Agent boxes
    agents = [
        ("SPATIAL", str(spatial_count), "RGB detections"),
        ("SPECTRAL", str(spectral_count), "Thermal regions"),
        ("FEATURE", str(feature_count), "Feature regions"),
        ("CONFLICT", str(candidate_count), "Candidates"),
        ("UNCERTAINTY", f"{average_confidence:.2f}", "Avg confidence")
    ]

    box_x = 35
    box_y = 110
    box_width = 215
    box_height = 120
    gap = 18

    for index, (
        name,
        value,
        description
    ) in enumerate(agents):

        x = (
            box_x +
            index * (box_width + gap)
        )

        cv2.rectangle(
            summary,
            (x, box_y),
            (x + box_width, box_y + box_height),
            (100, 100, 100),
            1
        )

        cv2.putText(
            summary,
            name,
            (x + 12, box_y + 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            summary,
            value,
            (x + 12, box_y + 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.82,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            summary,
            description,
            (x + 12, box_y + 99),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (180, 180, 180),
            1,
            cv2.LINE_AA
        )

    # Conflict
    cv2.putText(
        summary,
        "CONFLICT ANALYSIS",
        (35, 285),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    conflict_lines = [
        f"Candidates: {candidate_count}",
        f"Low conflict: {conflict_states['LOW']}",
        f"Moderate conflict: {conflict_states['MODERATE']}",
        f"High conflict: {conflict_states['HIGH']}",
        
    ]

    y = 320

    for line in conflict_lines:

        if line:
            cv2.putText(
                summary,
                line,
                (35, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                (220, 220, 220),
                1,
                cv2.LINE_AA
            )

        y += 30

    # Uncertainty
    cv2.putText(
        summary,
        "UNCERTAINTY ANALYSIS",
        (620, 285),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    uncertainty_lines = [
        f"Average uncertainty: {average_uncertainty:.3f}",
        f"Average confidence: {average_confidence:.3f}",
        "",
        f"High reliability: {reliability_counts['HIGH']}",
        f"Medium reliability: {reliability_counts['MEDIUM']}",
        f"Low reliability: {reliability_counts['LOW']}",
        
        
    ]

    y = 320

    for line in uncertainty_lines:

        if line:
            cv2.putText(
                summary,
                line,
                (620, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                (220, 220, 220),
                1,
                cv2.LINE_AA
            )

        y += 30

   
    output_path = OUTPUT_DIR / "review1_summary.png"

    cv2.imwrite(
        str(output_path),
        summary
    )

    return output_path


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("REVIEW 1 - MULTI-AGENT CAMOUFLAGE PIPELINE")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    captures = find_captures()

    if not captures:

        raise FileNotFoundError(
            "No complete RGB + Thermal captures found."
        )

    index_path = save_capture_index(captures)

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Capture number from capture_index.txt"
    )

    args = parser.parse_args()

    selected, selected_index = select_capture(
        captures,
        args.index
    )

    rgb_path = selected["rgb"]
    lwir_path = selected["lwir"]
    capture_dir = selected["directory"]

    print("\nSelected capture:")
    print(f"  Number: {selected_index}")
    print(f"  Directory: {capture_dir}")
    print(f"  RGB: {rgb_path}")
    print(f"  Thermal: {lwir_path}")

    save_inputs(
        rgb_path,
        lwir_path
    )

    with open(
        OUTPUT_DIR / "selected_capture.txt",
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            f"Capture number: {selected_index}\n"
        )
        file.write(
            f"Capture directory: {capture_dir}\n"
        )
        file.write(
            f"RGB: {rgb_path}\n"
        )
        file.write(
            f"Thermal: {lwir_path}\n"
        )

    # ========================================================
    # LOAD AGENTS
    # ========================================================

    print("\nLoading agents...")

    print("  Loading Spatial Agent...")
    spatial_agent = SpatialAgent(
        model_path=str(MODEL_PATH)
    )

    print("  Loading Spectral Agent...")
    spectral_agent = SpectralAgent()

    print("  Loading Feature Agent...")
    feature_agent = FeatureAgent()

    print("  Loading Conflict Agent...")
    conflict_agent = ConflictAgent()

    print("  Loading Uncertainty Agent...")
    uncertainty_agent = UncertaintyAgent()

    # ========================================================
    # 1. SPATIAL
    # ========================================================

    print("\n" + "-" * 70)
    print("1/5  SPATIAL AGENT")
    print("-" * 70)

    spatial_result = spatial_agent.analyze(
        rgb_path
    )

    print(
        f"Detections: "
        f"{len(spatial_result.get('detections', []))}"
    )

    spatial_agent.save_visualization(
        rgb_path,
        output_path=OUTPUT_DIR / "03_spatial_detection.png"
    )

    # ========================================================
    # 2. SPECTRAL
    # ========================================================

    print("\n" + "-" * 70)
    print("2/5  SPECTRAL AGENT")
    print("-" * 70)

    spectral_result = spectral_agent.analyze(
        lwir_path
    )

    print(
        f"Thermal regions: "
        f"{len(spectral_result.get('regions', []))}"
    )

    print(
        f"Combined thermal evidence: "
        f"{spectral_result.get('combined_evidence', 0.0):.3f}"
    )

    spectral_agent.save_visualization(
        lwir_path,
        output_path=OUTPUT_DIR / "04_spectral_analysis.png"
    )

    # ========================================================
    # 3. FEATURE
    # ========================================================

    print("\n" + "-" * 70)
    print("3/5  FEATURE AGENT")
    print("-" * 70)

    feature_result = feature_agent.analyze(
        rgb_path=rgb_path,
        lwir_path=lwir_path
    )

    print(
        f"Feature regions: "
        f"{len(feature_result.get('regions', []))}"
    )

    print(
        f"Patches scanned: "
        f"{feature_result.get('num_patches', 0)}"
    )

    print(
        f"Overall feature evidence: "
        f"{feature_result.get('overall_evidence', 0.0):.3f}"
    )

    feature_agent.save_visualization(
        rgb_path=rgb_path,
        lwir_path=lwir_path,
        output_path=OUTPUT_DIR / "05_feature_analysis.png"
    )

    # ========================================================
    # 4. CONFLICT
    # ========================================================

    print("\n" + "-" * 70)
    print("4/5  CONFLICT AGENT")
    print("-" * 70)

    conflict_result = conflict_agent.analyze(
        spatial_result,
        spectral_result,
        feature_result
    )

    conflict_candidates = conflict_result.get(
        "candidates",
        []
    )

    print(
        f"Candidates analysed: "
        f"{len(conflict_candidates)}"
    )

    for index, candidate in enumerate(
        conflict_candidates,
        start=1
    ):

        print(
            f"  C{index}: "
            f"{candidate.get('evidence_pattern', 'UNKNOWN')} | "
            f"Conflict: "
            f"{candidate.get('conflict_state', 'UNKNOWN')}"
        )

    conflict_agent.save_visualization(
        rgb_path,
        conflict_result,
        output_path=OUTPUT_DIR / "06_conflict_analysis.png"
    )

    # ========================================================
    # 5. UNCERTAINTY
    # ========================================================

    print("\n" + "-" * 70)
    print("5/5  UNCERTAINTY AGENT")
    print("-" * 70)

    uncertainty_results = uncertainty_agent.analyze(
        conflict_result
    )

    print(
        f"Candidates analysed: "
        f"{len(uncertainty_results)}"
    )

    for index, candidate in enumerate(
        uncertainty_results,
        start=1
    ):

        print(
            f"  C{index}: "
            f"Uncertainty "
            f"{candidate.get('uncertainty', 0.0):.3f} | "
            f"Confidence "
            f"{candidate.get('confidence', 0.0):.3f} | "
            f"Reliability "
            f"{candidate.get('reliability', 'LOW')}"
        )

    uncertainty_agent.save_visualization(
        rgb_path,
        uncertainty_results,
        output_path=OUTPUT_DIR / "07_uncertainty_analysis.png"
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "-" * 70)
    print("GENERATING REVIEW 1 SUMMARY")
    print("-" * 70)

    summary_path = create_summary(
        selected_index,
        capture_dir,
        spatial_result,
        spectral_result,
        feature_result,
        conflict_result,
        uncertainty_results
    )

    print(
        f"Summary saved to: "
        f"{summary_path}"
    )

    print("\n")
    print("=" * 70)
    print("REVIEW 1 PIPELINE COMPLETE")
    print("=" * 70)

    print(
        f"\nCapture used: #{selected_index}"
    )

    print(
        f"All outputs are in:\n{OUTPUT_DIR}"
    )

    print(
        f"\nCapture index:\n{index_path}"
    )


if __name__ == "__main__":
    main()
