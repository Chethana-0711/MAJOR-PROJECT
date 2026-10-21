from pathlib import Path
import argparse
import shutil
import json

import cv2
import numpy as np

from agents.spatial_agent import SpatialAgent
from agents.spectral_agent import SpectralAgent
from agents.feature_agent import FeatureAgent
from agents.conflict_agent import ConflictAgent
from agents.uncertainty_agent import UncertaintyAgent

# The older rule-based cognitive agent is optional: if agents/cognitive_agent.py
# is not present, the pipeline still runs with the Bayesian cognitive layer only.
try:
    from agents.cognitive_agent import CognitiveAgent as RuleCognitiveAgent
except ImportError:
    RuleCognitiveAgent = None

from agents.cognitive import (
    CognitiveAgent as BayesCognitiveAgent,
    analyze_candidates,
    to_legacy_candidate,
)
from agents.fusion import FusionAgent, SimpleFusionAgent

# Learned fusion (Method 1) is optional: used only when a trained model file exists.
try:
    from agents.learned_fusion.learned_fusion_agent import LearnedFusionAgent
except ImportError:
    LearnedFusionAgent = None


# ============================================================
# REVIEW 1 MULTI-AGENT PIPELINE
# ============================================================

PROJECT_DIR = Path(".")
DATASET_DIR = PROJECT_DIR / "MUDCAD-X" / "dataset"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "review1"

# Master capture-level split
SPLIT_FILE = PROJECT_DIR / "data" / "capture_splits.json"

MODEL_PATH = (
    PROJECT_DIR / "runs" / "detect" / "train" / "weights" / "best.pt"
)

# Trained by: python -m agents.learned_fusion.train --data data/fusion/val.csv
LEARNED_FUSION_MODEL = PROJECT_DIR / "models" / "learned_fusion.joblib"

# Which fusion layer decides the final output:
#   "simple"   - detectors carry the weight, cognitive layer smallest (recommended to start)
#   "weighted" - reliability-weighted fusion of all layers
#   "learned"  - trained meta-classifier (needs models/learned_fusion.joblib)
FUSION_METHOD = "simple"


# ============================================================
# LOAD MASTER TEST SPLIT
# ============================================================

def load_test_split():

    if not SPLIT_FILE.exists():

        raise FileNotFoundError(
            f"Master split file not found:\n"
            f"{SPLIT_FILE}\n\n"
            f"Please create data/capture_splits.json first."
        )

    with open(
        SPLIT_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        split_data = json.load(file)

    if "test" not in split_data:

        raise KeyError(
            "The master split file does not contain a 'test' split."
        )

    test_captures = split_data["test"]

    if not test_captures:

        raise ValueError(
            "The master test split is empty."
        )

    return test_captures


# ============================================================
# FIND TEST CAPTURES
# ============================================================

def find_captures():

    test_split = load_test_split()

    captures = []

    for capture_entry in test_split:

        # The split file stores the capture directory.
        capture_dir = Path(
            capture_entry
        )

        # If the path in the JSON is relative,
        # interpret it relative to the project root.
        if not capture_dir.is_absolute():

            capture_dir = PROJECT_DIR / capture_dir

        rgb_path = capture_dir / "vis.png"
        lwir_path = capture_dir / "lwir.png"

        if not rgb_path.exists():

            print(
                f"WARNING: RGB image missing:\n"
                f"  {rgb_path}"
            )

            continue

        if not lwir_path.exists():

            print(
                f"WARNING: Thermal image missing:\n"
                f"  {lwir_path}"
            )

            continue

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

        file.write(
            "REVIEW 1 - HELD-OUT TEST CAPTURE INDEX\n"
        )

        file.write(
            "=" * 80 + "\n\n"
        )

        file.write(
            f"Number of test captures: {len(captures)}\n"
        )

        file.write(
            "Source: data/capture_splits.json\n"
        )

        file.write(
            "Split: TEST\n\n"
        )

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

def select_capture(
    captures,
    requested_index=None
):

    if requested_index is not None:

        if not 1 <= requested_index <= len(captures):

            raise ValueError(
                f"Test capture number must be between "
                f"1 and {len(captures)}."
            )

        return (
            captures[requested_index - 1],
            requested_index
        )

    print("\n")
    print("=" * 70)
    print(" TEST CAPTURE SELECTION")
    print("=" * 70)

    print(
        f"\nHeld-out TEST captures available: "
        f"{len(captures)}"
    )

    print(
        "\nCapture list saved to:"
    )

    print(
        f"  {OUTPUT_DIR / 'capture_index.txt'}"
    )

    print(
        "\nEnter a TEST capture number."
    )

    print(
        "Press ENTER to use test capture #1."
    )

    while True:

        user_input = input(
            "\nEnter test capture number: "
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

def save_inputs(
    rgb_path,
    lwir_path
):

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
    uncertainty_result,
    cognitive_results=None,
    fusion_results=None
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
    height = 800

    summary = np.zeros(
        (height, width, 3),
        dtype=np.uint8
    )

    # ========================================================
    # HEADER
    # ========================================================

    cv2.putText(
        summary,
        "MULTI-AGENT CAMOUFLAGE ANALYSIS",
        (35, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        summary,
        f"TEST Capture #{selected_index}  |  {capture_dir}",
        (35, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (190, 190, 190),
        1,
        cv2.LINE_AA
    )

    # ========================================================
    # AGENT BOXES
    # ========================================================

    agents = [
        (
            "SPATIAL",
            str(spatial_count),
            "RGB detections"
        ),
        (
            "SPECTRAL",
            str(spectral_count),
            "Thermal regions"
        ),
        (
            "FEATURE",
            str(feature_count),
            "Feature regions"
        ),
        (
            "CONFLICT",
            str(candidate_count),
            "Candidates"
        ),
        (
            "UNCERTAINTY",
            f"{average_confidence:.2f}",
            "Avg confidence"
        )
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

    # ========================================================
    # CONFLICT ANALYSIS
    # ========================================================

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
        f"High conflict: {conflict_states['HIGH']}"
    ]

    y = 320

    for line in conflict_lines:

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

    # ========================================================
    # UNCERTAINTY ANALYSIS
    # ========================================================

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
        f"Low reliability: {reliability_counts['LOW']}"
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

    # ========================================================
    # COGNITIVE ANALYSIS
    # ========================================================

    cognitive_results = cognitive_results or []

    hyp_counts = {"H0": 0, "H1": 0, "H2": 0, "REVIEW": 0}

    for candidate in cognitive_results:

        hypothesis = candidate.get("bayes_hypothesis")

        if hypothesis in hyp_counts:
            hyp_counts[hypothesis] += 1
        else:
            hyp_counts["REVIEW"] += 1

    average_suspicion = (
        float(np.mean([
            candidate.get("bayes_suspicion", 0.0)
            for candidate in cognitive_results
        ]))
        if cognitive_results
        else 0.0
    )

    cv2.putText(
        summary,
        "COGNITIVE ANALYSIS",
        (35, 525),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cognitive_lines = [
        f"Camouflaged object (H0): {hyp_counts['H0']}",
        f"Ordinary object (H1): {hyp_counts['H1']}",
        f"No object (H2): {hyp_counts['H2']}",
        f"Under review: {hyp_counts['REVIEW']}",
        f"Average suspicion P(H0): {average_suspicion:.3f}"
    ]

    y = 560

    for line in cognitive_lines:

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

    cv2.putText(
        summary,
        "CANDIDATE DECISIONS",
        (620, 525),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    y = 560

    for index, candidate in enumerate(
        cognitive_results[:7],
        start=1
    ):

        decision = candidate.get(
            "bayes_decision",
            "UNKNOWN"
        )

        suspicion = candidate.get(
            "bayes_suspicion",
            0.0
        )

        cv2.putText(
            summary,
            f"C{index}: {decision}  (P(H0)={suspicion:.2f})",
            (620, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (220, 220, 220),
            1,
            cv2.LINE_AA
        )

        y += 30

    # ========================================================
    # FINAL OUTPUT (FUSION)
    # ========================================================

    fusion_results = fusion_results or []

    final_counts = {
        "CAMOUFLAGE DETECTED": 0,
        "NOT DETECTED": 0,
        "REVIEW": 0
    }

    for candidate in fusion_results:

        status = candidate.get("final_status", "REVIEW")

        if status in final_counts:
            final_counts[status] += 1

    cv2.putText(
        summary,
        (
            f"FINAL (FUSION)  Detected: "
            f"{final_counts['CAMOUFLAGE DETECTED']}   "
            f"Not detected: {final_counts['NOT DETECTED']}   "
            f"Review: {final_counts['REVIEW']}"
        ),
        (35, 770),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    output_path = (
        OUTPUT_DIR /
        "review1_summary.png"
    )

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
    
    print("MULTI-AGENT CAMOUFLAGE PIPELINE")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # LOAD ONLY HELD-OUT TEST CAPTURES
    # ========================================================

    print("\nLoading master capture split...")

    captures = find_captures()

    if not captures:

        raise FileNotFoundError(
            "No complete RGB + Thermal TEST captures found."
        )

    print(
        f"TEST captures available: {len(captures)}"
    )

    if len(captures) != 129:

        print(
            "\nWARNING:"
        )

        print(
            "Expected 129 test captures from the "
            "master split."
        )

        print(
            f"Found {len(captures)} usable test captures."
        )

    index_path = save_capture_index(
        captures
    )

    # ========================================================
    # ARGUMENTS
    # ========================================================

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help=(
            "Test capture number from "
            "capture_index.txt"
        )
    )

    args = parser.parse_args()

    selected, selected_index = select_capture(
        captures,
        args.index
    )

    rgb_path = selected["rgb"]
    lwir_path = selected["lwir"]
    capture_dir = selected["directory"]

    # ========================================================
    # SELECTED CAPTURE
    # ========================================================

    print("\nSelected TEST capture:")

    print(
        f"  Test number: {selected_index}"
    )

    print(
        f"  Directory: {capture_dir}"
    )

    print(
        f"  RGB: {rgb_path}"
    )

    print(
        f"  Thermal: {lwir_path}"
    )

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
            "REVIEW 1 SELECTED CAPTURE\n"
        )

        file.write(
            "=" * 60 + "\n"
        )

        file.write(
            "Split: TEST\n"
        )

        file.write(
            f"Test capture number: {selected_index}\n"
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

    print(
        "  Loading Spatial Agent..."
    )

    spatial_agent = SpatialAgent(
        model_path=str(MODEL_PATH)
    )

    print(
        "  Loading Spectral Agent..."
    )

    spectral_agent = SpectralAgent()

    print(
        "  Loading Feature Agent..."
    )

    feature_agent = FeatureAgent()

    print(
        "  Loading Conflict Agent..."
    )

    conflict_agent = ConflictAgent()

    print(
        "  Loading Uncertainty Agent..."
    )

    uncertainty_agent = UncertaintyAgent()

    # ========================================================
    # 1. SPATIAL
    # ========================================================

    
    print("1/7  SPATIAL AGENT")
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
        output_path=(
            OUTPUT_DIR /
            "03_spatial_detection.png"
        )
    )

    # ========================================================
    # 2. SPECTRAL
    # ========================================================

    print("\n")
    print("2/7  SPECTRAL AGENT")
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
        output_path=(
            OUTPUT_DIR /
            "04_spectral_analysis.png"
        )
    )

    # ========================================================
    # 3. FEATURE
    # ========================================================

    print("\n")
    print("3/7  FEATURE AGENT")
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
        output_path=(
            OUTPUT_DIR /
            "05_feature_analysis.png"
        )
    )

    # ========================================================
    # 4. CONFLICT
    # ========================================================

    print("\n")
    print("4/7  CONFLICT AGENT")
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
        output_path=(
            OUTPUT_DIR /
            "06_conflict_analysis.png"
        )
    )

    # ========================================================
    # 5. UNCERTAINTY
    # ========================================================

    print("\n")
    print("5/7  UNCERTAINTY AGENT")
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
        output_path=(
            OUTPUT_DIR /
            "07_uncertainty_analysis.png"
        )
    )

    # ========================================================
    # 6. COGNITIVE
    # ========================================================

    print("\n")
    print("6/7  COGNITIVE AGENT")
    print("-" * 70)

    bayes_agent = BayesCognitiveAgent()
    rule_agent = (
        RuleCognitiveAgent()
        if RuleCognitiveAgent is not None
        else None
    )

    if rule_agent is None:

        print(
            "  (agents/cognitive_agent.py not found - "
            "running Bayesian cognitive layer only)"
        )

    cognitive_results = analyze_candidates(
        bayes_agent,
        uncertainty_results
    )

    for index, candidate in enumerate(
        cognitive_results,
        start=1
    ):

        if rule_agent is not None:

            baseline = rule_agent.reason(
                to_legacy_candidate(candidate)
            )

            candidate["rule_decision"] = baseline["decision"]
            candidate["rule_confidence"] = baseline["confidence"]

        else:

            candidate["rule_decision"] = "N/A"
            candidate["rule_confidence"] = 0.0

        posterior = candidate["bayes_posterior"]

        print(
            f"  C{index}: "
            f"Bayesian: {candidate['bayes_decision']} | "
            f"P(H0)={posterior['H0']:.2f} "
            f"P(H1)={posterior['H1']:.2f} "
            f"P(H2)={posterior['H2']:.2f} | "
            f"Rule-based: {candidate['rule_decision']}"
        )

        for line in candidate["bayes_explanation"]:
            print(f"        - {line}")

    cognitive_path = OUTPUT_DIR / "08_cognitive_results.json"

    with open(
        cognitive_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            cognitive_results,
            file,
            indent=2,
            default=str
        )

    print(
        f"Cognitive results saved to: "
        f"{cognitive_path}"
    )

    # ========================================================
    # 7. FUSION
    # ========================================================

    print("\n")
    print("7/7  FUSION AGENT")
    print("-" * 70)

    if (
        FUSION_METHOD == "learned"
        and LearnedFusionAgent is not None
        and LEARNED_FUSION_MODEL.exists()
    ):

        fusion_agent = LearnedFusionAgent(
            str(LEARNED_FUSION_MODEL)
        )

        print("  Method: LEARNED fusion (trained meta-classifier)")

    elif FUSION_METHOD == "weighted":

        fusion_agent = FusionAgent()

        print("  Method: WEIGHTED fusion (all layers)")

    else:

        fusion_agent = SimpleFusionAgent()

        print(
            "  Method: SIMPLE fusion "
            "(spatial/spectral/feature weighted most, cognitive least)"
        )

    fusion_results = fusion_agent.analyze(
        cognitive_results
    )

    cleared = 0

    for index, candidate in enumerate(
        fusion_results,
        start=1
    ):

        if candidate["final_status"] == "NOT DETECTED":
            cleared += 1
            continue

        signals = candidate["fusion_signals"]

        print(
            f"  C{index}: {candidate['final_status']} | "
            f"{candidate['final_label']} "
            f"({candidate['final_sublabel'] or '-'}) | "
            f"Suspicion {candidate['final_suspicion']:.2f}"
        )

        print(f"        signals: {signals}")

        for reason in candidate["fusion_reasons"]:
            print(f"        - {reason}")

    print(f"  ({cleared} candidates cleared as NOT DETECTED)")

    fusion_path = OUTPUT_DIR / "09_fusion_results.json"

    with open(
        fusion_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            fusion_results,
            file,
            indent=2,
            default=str
        )

    # all candidates (debug view)
    fusion_agent.save_visualization(
        rgb_path,
        fusion_results,
        output_path=(
            OUTPUT_DIR /
            "10b_all_candidates.png"
        )
    )

    # FINAL OUTPUT: only camouflaged objects, with the evidence behind each one
    fusion_agent.save_certified_output(
        rgb_path,
        fusion_results,
        output_path=(
            OUTPUT_DIR /
            "10_final_output.png"
        ),
        report_path=(
            OUTPUT_DIR /
            "11_camouflaged_objects.json"
        )
    )

    print(
        f"Fusion results saved to: "
        f"{fusion_path}"
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n")
    print("SUMMARY")
    print("-" * 70)

    summary_path = create_summary(
        selected_index,
        capture_dir,
        spatial_result,
        spectral_result,
        feature_result,
        conflict_result,
        uncertainty_results,
        cognitive_results,
        fusion_results
    )

    print(
        f"Summary saved to: "
        f"{summary_path}"
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n")
    
    print("PIPELINE COMPLETE")
    print("=" * 70)

    print(
        f"\nTEST capture used: #{selected_index}"
    )

    print(
        f"Total TEST captures: {len(captures)}"
    )

    print(
        f"All outputs are in:\n{OUTPUT_DIR}"
    )

    print(
        f"\nTEST capture index:\n{index_path}"
    )


if __name__ == "__main__":
    main()