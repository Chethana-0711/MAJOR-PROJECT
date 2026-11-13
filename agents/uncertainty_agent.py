from pathlib import Path

import cv2
import numpy as np


# ============================================================
# UNCERTAINTY AGENT
# ============================================================

class UncertaintyAgent:
    """
    Uncertainty Agent

    Estimates confidence/reliability for each candidate
    produced by the Conflict Agent.

    The agent combines:

        1. Coverage uncertainty
        2. Evidence uncertainty
        3. Conflict uncertainty

    Important:

        NOT_AVAILABLE
            -> excluded from evidence calculation

        NO_DETECTION
            -> contributes zero evidence

        DETECTED
            -> contributes its evidence value

        EVIDENCE_PRESENT
            -> contributes its localized evidence value
    """

    # --------------------------------------------------------
    # WEIGHTS
    # --------------------------------------------------------

    COVERAGE_WEIGHT = 0.30
    EVIDENCE_WEIGHT = 0.40
    CONFLICT_WEIGHT = 0.30

    # --------------------------------------------------------
    # RELIABILITY THRESHOLDS
    # --------------------------------------------------------

    HIGH_RELIABILITY_THRESHOLD = 0.75
    MEDIUM_RELIABILITY_THRESHOLD = 0.50

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(self):

        pass

    # ========================================================
    # GET EVIDENCE
    # ========================================================

    def _get_agent_evidence(
        self,
        status,
        evidence
    ):
        """
        Convert an agent status/evidence pair into a usable
        uncertainty-analysis value.

        NOT_AVAILABLE is represented by None so it can be
        excluded from the evidence average.
        """

        if status == "NOT_AVAILABLE":

            return None

        if status == "NO_DETECTION":

            return 0.0

        if status in [
            "DETECTED",
            "EVIDENCE_PRESENT"
        ]:

            return float(
                np.clip(
                    evidence,
                    0.0,
                    1.0
                )
            )

        return None

    # ========================================================
    # ANALYZE ONE CANDIDATE
    # ========================================================

    def _analyze_candidate(
        self,
        candidate
    ):
        """
        Calculate uncertainty for one Conflict Agent
        candidate.
        """

        # ----------------------------------------------------
        # AGENT STATUS
        # ----------------------------------------------------

        spatial_status = candidate.get(
            "spatial_status",
            "NOT_AVAILABLE"
        )

        spectral_status = candidate.get(
            "spectral_status",
            "NOT_AVAILABLE"
        )

        feature_status = candidate.get(
            "feature_status",
            "NOT_AVAILABLE"
        )

        # ----------------------------------------------------
        # AGENT EVIDENCE
        # ----------------------------------------------------

        spatial_evidence = float(
            candidate.get(
                "spatial_evidence",
                0.0
            )
        )

        spectral_evidence = float(
            candidate.get(
                "spectral_evidence",
                0.0
            )
        )

        feature_evidence = float(
            candidate.get(
                "feature_evidence",
                0.0
            )
        )

        # ----------------------------------------------------
        # AVAILABLE AGENTS
        # ----------------------------------------------------

        available_agents = candidate.get(
            "available_agents",
            []
        )

        available_count = int(
            candidate.get(
                "available_count",
                len(available_agents)
            )
        )

        # Safety clamp

        available_count = max(
            0,
            min(
                available_count,
                3
            )
        )

        # ----------------------------------------------------
        # SUPPORTING AGENTS
        # ----------------------------------------------------

        supporting_agents = candidate.get(
            "supporting_agents",
            []
        )

        support_count = int(
            candidate.get(
                "support_count",
                len(supporting_agents)
            )
        )

        # ----------------------------------------------------
        # CONFLICT
        # ----------------------------------------------------

        conflict = float(
            np.clip(
                candidate.get(
                    "conflict",
                    0.0
                ),
                0.0,
                1.0
            )
        )

        agreement = float(
            np.clip(
                candidate.get(
                    "agreement",
                    support_count / 3.0
                    if support_count >= 0
                    else 0.0
                ),
                0.0,
                1.0
            )
        )

        # ====================================================
        # EVIDENCE VALUES
        # ====================================================

        evidence_values = []

        spatial_value = self._get_agent_evidence(
            spatial_status,
            spatial_evidence
        )

        if spatial_value is not None:

            evidence_values.append(
                spatial_value
            )

        spectral_value = self._get_agent_evidence(
            spectral_status,
            spectral_evidence
        )

        if spectral_value is not None:

            evidence_values.append(
                spectral_value
            )

        feature_value = self._get_agent_evidence(
            feature_status,
            feature_evidence
        )

        if feature_value is not None:

            evidence_values.append(
                feature_value
            )

        # ====================================================
        # EVIDENCE STRENGTH
        # ====================================================

        if evidence_values:

            evidence_strength = float(
                np.mean(
                    evidence_values
                )
            )

        else:

            evidence_strength = 0.0

        evidence_strength = float(
            np.clip(
                evidence_strength,
                0.0,
                1.0
            )
        )

        # ====================================================
        # COVERAGE UNCERTAINTY
        # ====================================================

        coverage_uncertainty = (
            1.0 -
            (
                available_count /
                3.0
            )
        )

        coverage_uncertainty = float(
            np.clip(
                coverage_uncertainty,
                0.0,
                1.0
            )
        )

        # ====================================================
        # EVIDENCE UNCERTAINTY
        # ====================================================

        evidence_uncertainty = (
            1.0 -
            evidence_strength
        )

        evidence_uncertainty = float(
            np.clip(
                evidence_uncertainty,
                0.0,
                1.0
            )
        )

        # ====================================================
        # CONFLICT UNCERTAINTY
        # ====================================================

        conflict_uncertainty = conflict

        # ====================================================
        # OVERALL UNCERTAINTY
        # ====================================================

        overall_uncertainty = (

            self.COVERAGE_WEIGHT *
            coverage_uncertainty

            +

            self.EVIDENCE_WEIGHT *
            evidence_uncertainty

            +

            self.CONFLICT_WEIGHT *
            conflict_uncertainty
        )

        overall_uncertainty = float(
            np.clip(
                overall_uncertainty,
                0.0,
                1.0
            )
        )

        # ====================================================
        # CONFIDENCE
        # ====================================================

        confidence = (
            1.0 -
            overall_uncertainty
        )

        confidence = float(
            np.clip(
                confidence,
                0.0,
                1.0
            )
        )

        # ====================================================
        # RELIABILITY
        # ====================================================

        if (
            confidence >=
            self.HIGH_RELIABILITY_THRESHOLD
        ):

            reliability = "HIGH"

        elif (
            confidence >=
            self.MEDIUM_RELIABILITY_THRESHOLD
        ):

            reliability = "MEDIUM"

        else:

            reliability = "LOW"

        # ====================================================
        # RESULT
        # ====================================================

        return {

            "bbox":
                candidate.get(
                    "bbox",
                    [0, 0, 0, 0]
                ),

            "class":
                candidate.get(
                    "class",
                    "unknown"
                ),

            # ----------------------------------------------
            # STATUS
            # ----------------------------------------------

            "spatial_status":
                spatial_status,

            "spectral_status":
                spectral_status,

            "feature_status":
                feature_status,

            # ----------------------------------------------
            # EVIDENCE
            # ----------------------------------------------

            "spatial_evidence":
                spatial_evidence,

            "spectral_evidence":
                spectral_evidence,

            "feature_evidence":
                feature_evidence,

            # ----------------------------------------------
            # AVAILABILITY
            # ----------------------------------------------

            "available_agents":
                available_agents,

            "available_count":
                available_count,

            # ----------------------------------------------
            # SUPPORT
            # ----------------------------------------------

            "supporting_agents":
                supporting_agents,

            "support_count":
                support_count,

            # ----------------------------------------------
            # CONFLICT
            # ----------------------------------------------

            "agreement":
                agreement,

            "conflict":
                conflict,

            "conflict_state":
                candidate.get(
                    "conflict_state",
                    "UNKNOWN"
                ),

            "evidence_pattern":
                candidate.get(
                    "evidence_pattern",
                    "UNKNOWN"
                ),

            # ----------------------------------------------
            # UNCERTAINTY
            # ----------------------------------------------

            "evidence_strength":
                evidence_strength,

            "coverage_uncertainty":
                coverage_uncertainty,

            "evidence_uncertainty":
                evidence_uncertainty,

            "conflict_uncertainty":
                conflict_uncertainty,

            "uncertainty":
                overall_uncertainty,

            # ----------------------------------------------
            # CONFIDENCE
            # ----------------------------------------------

            "confidence":
                confidence,

            "reliability":
                reliability
        }

    # ========================================================
    # MAIN ANALYSIS
    # ========================================================

    def analyze(
        self,
        conflict_result
    ):
        """
        Analyze candidates produced by the Conflict Agent.

        Accepts either:

            conflict_result["candidates"]

        or directly:

            candidates
        """

        # ----------------------------------------------------
        # Accept dictionary or candidate list
        # ----------------------------------------------------

        if isinstance(
            conflict_result,
            dict
        ):

            candidates = conflict_result.get(
                "candidates",
                []
            )

        elif isinstance(
            conflict_result,
            list
        ):

            candidates = conflict_result

        else:

            candidates = []

        # ----------------------------------------------------
        # Analyze candidates
        # ----------------------------------------------------

        results = []

        for candidate in candidates:

            results.append(
                self._analyze_candidate(
                    candidate
                )
            )

        return results

    # ========================================================
    # VISUALIZATION
    # ========================================================

    def save_visualization(
        self,
        rgb_path,
        results,
        output_path="outputs/uncertainty_analysis.png"
    ):
        """
        Presentation-ready Uncertainty Agent visualization.

        Left:
            RGB image with candidate boxes.

        Right:
            Compact uncertainty statistics.

        Box colors represent reliability:

            GREEN  = HIGH
            YELLOW = MEDIUM
            RED    = LOW
        """

        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        # ====================================================
        # LOAD RGB
        # ====================================================

        image = cv2.imread(
            str(rgb_path)
        )

        if image is None:

            raise ValueError(
                f"Could not load RGB image: "
                f"{rgb_path}"
            )

        # Make a copy so the input is not modified.

        display_image = image.copy()

        # ====================================================
        # DRAW CANDIDATES
        # ====================================================

        for index, candidate in enumerate(
            results,
            start=1
        ):

            bbox = candidate.get(
                "bbox",
                [0, 0, 0, 0]
            )

            x, y, w, h = [
                int(
                    round(
                        float(value)
                    )
                )
                for value in bbox
            ]

            # ----------------------------------------------
            # Clamp coordinates
            # ----------------------------------------------

            x = max(
                0,
                min(
                    x,
                    display_image.shape[1] - 1
                )
            )

            y = max(
                0,
                min(
                    y,
                    display_image.shape[0] - 1
                )
            )

            w = max(
                1,
                min(
                    w,
                    display_image.shape[1] - x
                )
            )

            h = max(
                1,
                min(
                    h,
                    display_image.shape[0] - y
                )
            )

            # ----------------------------------------------
            # RELIABILITY COLOR
            # ----------------------------------------------

            reliability = candidate.get(
                "reliability",
                "LOW"
            )

            if reliability == "HIGH":

                # Green

                color = (
                    0,
                    200,
                    0
                )

            elif reliability == "MEDIUM":

                # Yellow

                color = (
                    0,
                    220,
                    220
                )

            else:

                # Red

                color = (
                    0,
                    0,
                    255
                )

            # ----------------------------------------------
            # BOX
            # ----------------------------------------------

            cv2.rectangle(
                display_image,
                (x, y),
                (
                    x + w,
                    y + h
                ),
                color,
                2
            )

            # ----------------------------------------------
            # LABEL
            # ----------------------------------------------

            cv2.putText(
                display_image,
                f"C{index}",
                (
                    x,
                    max(
                        y - 8,
                        20
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA
            )

        # ====================================================
        # STATISTICS
        # ====================================================

        candidate_count = len(
            results
        )

        uncertainty_values = [
            float(
                candidate.get(
                    "uncertainty",
                    0.0
                )
            )
            for candidate in results
        ]

        confidence_values = [
            float(
                candidate.get(
                    "confidence",
                    0.0
                )
            )
            for candidate in results
        ]

        if uncertainty_values:

            average_uncertainty = float(
                np.mean(
                    uncertainty_values
                )
            )

        else:

            average_uncertainty = 0.0

        if confidence_values:

            average_confidence = float(
                np.mean(
                    confidence_values
                )
            )

        else:

            average_confidence = 0.0

        # ----------------------------------------------------
        # Reliability counts
        # ----------------------------------------------------

        reliability_counts = {
            "HIGH": 0,
            "MEDIUM": 0,
            "LOW": 0
        }

        for candidate in results:

            reliability = candidate.get(
                "reliability",
                "LOW"
            )

            if reliability in reliability_counts:

                reliability_counts[
                    reliability
                ] += 1

        # ====================================================
        # PANEL
        # ====================================================

        panel_width = 430

        panel_height = display_image.shape[0]

        panel = np.zeros(
            (
                panel_height,
                panel_width,
                3
            ),
            dtype=np.uint8
        )

        # ====================================================
        # HEADER
        # ====================================================

        cv2.putText(
            panel,
            "UNCERTAINTY AGENT",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            panel,
            f"Candidates: {candidate_count}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (220, 220, 220),
            1,
            cv2.LINE_AA
        )

        # ====================================================
        # UNCERTAINTY
        # ====================================================

        cv2.putText(
            panel,
            "UNCERTAINTY",
            (20, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            panel,
            (
                f"Average: "
                f"{average_uncertainty:.3f}"
            ),
            (25, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (220, 220, 220),
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            panel,
            (
                f"Confidence: "
                f"{average_confidence:.3f}"
            ),
            (25, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (220, 220, 220),
            1,
            cv2.LINE_AA
        )

        # ====================================================
        # RELIABILITY
        # ====================================================

        cv2.putText(
            panel,
            "RELIABILITY",
            (20, 230),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        reliability_lines = [
            (
                "High: "
                f"{reliability_counts['HIGH']}"
            ),
            (
                "Medium: "
                f"{reliability_counts['MEDIUM']}"
            ),
            (
                "Low: "
                f"{reliability_counts['LOW']}"
            )
        ]

        y_position = 265

        for line in reliability_lines:

            cv2.putText(
                panel,
                line,
                (25, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (220, 220, 220),
                1,
                cv2.LINE_AA
            )

            y_position += 30

        # ====================================================
        # UNCERTAINTY COMPONENTS
        # ====================================================

        if results:

            coverage_values = [
                candidate.get(
                    "coverage_uncertainty",
                    0.0
                )
                for candidate in results
            ]

            evidence_values = [
                candidate.get(
                    "evidence_uncertainty",
                    0.0
                )
                for candidate in results
            ]

            conflict_values = [
                candidate.get(
                    "conflict_uncertainty",
                    0.0
                )
                for candidate in results
            ]

            average_coverage = float(
                np.mean(
                    coverage_values
                )
            )

            average_evidence_uncertainty = float(
                np.mean(
                    evidence_values
                )
            )

            average_conflict_uncertainty = float(
                np.mean(
                    conflict_values
                )
            )

        else:

            average_coverage = 0.0
            average_evidence_uncertainty = 0.0
            average_conflict_uncertainty = 0.0

        # ----------------------------------------------------
        # Components section
        # ----------------------------------------------------

        cv2.putText(
            panel,
            "UNCERTAINTY COMPONENTS",
            (20, 370),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        component_lines = [
            (
                "Coverage: "
                f"{average_coverage:.3f}"
            ),
            (
                "Evidence weakness: "
                f"{average_evidence_uncertainty:.3f}"
            ),
            (
                "Conflict: "
                f"{average_conflict_uncertainty:.3f}"
            )
        ]

        y_position = 405

        for line in component_lines:

            cv2.putText(
                panel,
                line,
                (25, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (220, 220, 220),
                1,
                cv2.LINE_AA
            )

            y_position += 28

        # ====================================================
        # COMBINE
        # ====================================================

        final_image = np.hstack([
            display_image,
            panel
        ])

        cv2.imwrite(
            str(output_path),
            final_image
        )

        print(
            f"Visualization saved to: "
            f"{output_path}"
        )

        return output_path


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    agent = UncertaintyAgent()

    test_candidates = [

        {
            "bbox": [
                100,
                100,
                80,
                80
            ],

            "class":
                "test",

            "spatial_status":
                "DETECTED",

            "spectral_status":
                "DETECTED",

            "feature_status":
                "DETECTED",

            "spatial_evidence":
                0.90,

            "spectral_evidence":
                0.85,

            "feature_evidence":
                0.80,

            "available_agents":
                [
                    "spatial",
                    "spectral",
                    "feature"
                ],

            "available_count":
                3,

            "supporting_agents":
                [
                    "spatial",
                    "spectral",
                    "feature"
                ],

            "support_count":
                3,

            "agreement":
                1.0,

            "conflict":
                0.10,

            "conflict_state":
                "LOW",

            "evidence_pattern":
                "THREE-AGENT AGREEMENT"
        }
    ]

    results = agent.analyze(
        test_candidates
    )

    print(
        "\nUNCERTAINTY AGENT"
    )

    print(
        "=" * 60
    )

    for index, result in enumerate(
        results,
        start=1
    ):

        print(
            f"\nCandidate {index}"
        )

        print(
            f"  Evidence strength: "
            f"{result['evidence_strength']:.3f}"
        )

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