"""Uncertainty Agent for the multimodal camouflage-detection pipeline.

The calculation logic and output schema are preserved from the supplied
implementation. The visualization is reorganized into small helper methods so
that drawing and statistics are easier to maintain.
"""

from pathlib import Path

import cv2
import numpy as np


class UncertaintyAgent:
    """Estimate candidate reliability from agent coverage, evidence, and conflict.

    Status handling:
      * NOT_AVAILABLE: exclude this agent from the evidence average.
      * NO_DETECTION: include zero evidence in the average.
      * DETECTED / EVIDENCE_PRESENT: include the clipped evidence value.

    The weighted uncertainty formula and reliability thresholds are intentionally
    unchanged. The visualization may look different from the original version.
    """

    # Keep these values identical to the supplied implementation.
    COVERAGE_WEIGHT = 0.30
    EVIDENCE_WEIGHT = 0.40
    CONFLICT_WEIGHT = 0.30

    HIGH_RELIABILITY_THRESHOLD = 0.75
    MEDIUM_RELIABILITY_THRESHOLD = 0.50

    AGENT_NAMES = ("spatial", "spectral", "feature")
    RELIABILITY_COLORS = {
        "HIGH": (0, 200, 0),
        "MEDIUM": (0, 220, 220),
        "LOW": (0, 0, 255),
    }

    def __init__(self):
        """Initialize the agent. No model state is required."""

    # ------------------------------------------------------------------
    # Evidence handling
    # ------------------------------------------------------------------

    @staticmethod
    def _get_agent_evidence(status, evidence):
        """Map one agent's status/evidence pair to a usable numeric value."""
        if status == "NOT_AVAILABLE":
            return None
        if status == "NO_DETECTION":
            return 0.0
        if status in ("DETECTED", "EVIDENCE_PRESENT"):
            return float(np.clip(evidence, 0.0, 1.0))
        return None

    def _analyze_candidate(self, candidate):
        """Calculate the uncertainty fields for a single conflict candidate."""
        # Read the three statuses using the same defaults as the original.
        spatial_status = candidate.get("spatial_status", "NOT_AVAILABLE")
        spectral_status = candidate.get("spectral_status", "NOT_AVAILABLE")
        feature_status = candidate.get("feature_status", "NOT_AVAILABLE")

        # Preserve the original float conversion and default values.
        spatial_evidence = float(candidate.get("spatial_evidence", 0.0))
        spectral_evidence = float(candidate.get("spectral_evidence", 0.0))
        feature_evidence = float(candidate.get("feature_evidence", 0.0))

        available_agents = candidate.get("available_agents", [])
        available_count = int(
            candidate.get("available_count", len(available_agents))
        )
        available_count = max(0, min(available_count, 3))

        supporting_agents = candidate.get("supporting_agents", [])
        support_count = int(
            candidate.get("support_count", len(supporting_agents))
        )

        conflict = float(np.clip(candidate.get("conflict", 0.0), 0.0, 1.0))
        agreement = float(
            np.clip(
                candidate.get(
                    "agreement",
                    support_count / 3.0 if support_count >= 0 else 0.0,
                ),
                0.0,
                1.0,
            )
        )

        # Build evidence in the same spatial/spectral/feature order.
        evidence_values = []
        for status, value in (
            (spatial_status, spatial_evidence),
            (spectral_status, spectral_evidence),
            (feature_status, feature_evidence),
        ):
            agent_value = self._get_agent_evidence(status, value)
            if agent_value is not None:
                evidence_values.append(agent_value)

        # Keep the original mean, empty-list fallback, and clipping.
        if evidence_values:
            evidence_strength = float(np.mean(evidence_values))
        else:
            evidence_strength = 0.0
        evidence_strength = float(np.clip(evidence_strength, 0.0, 1.0))

        # These component calculations are unchanged.
        coverage_uncertainty = 1.0 - (available_count / 3.0)
        coverage_uncertainty = float(
            np.clip(coverage_uncertainty, 0.0, 1.0)
        )

        evidence_uncertainty = 1.0 - evidence_strength
        evidence_uncertainty = float(
            np.clip(evidence_uncertainty, 0.0, 1.0)
        )

        conflict_uncertainty = conflict

        # Preserve the exact weighted formula and its clipping.
        overall_uncertainty = (
            self.COVERAGE_WEIGHT * coverage_uncertainty
            + self.EVIDENCE_WEIGHT * evidence_uncertainty
            + self.CONFLICT_WEIGHT * conflict_uncertainty
        )
        overall_uncertainty = float(
            np.clip(overall_uncertainty, 0.0, 1.0)
        )

        confidence = 1.0 - overall_uncertainty
        confidence = float(np.clip(confidence, 0.0, 1.0))

        # Keep the original threshold comparisons, including equality.
        if confidence >= self.HIGH_RELIABILITY_THRESHOLD:
            reliability = "HIGH"
        elif confidence >= self.MEDIUM_RELIABILITY_THRESHOLD:
            reliability = "MEDIUM"
        else:
            reliability = "LOW"

        # Preserve every key consumed by downstream pipeline stages.
        return {
            "bbox": candidate.get("bbox", [0, 0, 0, 0]),
            "class": candidate.get("class", "unknown"),
            "spatial_status": spatial_status,
            "spectral_status": spectral_status,
            "feature_status": feature_status,
            "spatial_evidence": spatial_evidence,
            "spectral_evidence": spectral_evidence,
            "feature_evidence": feature_evidence,
            "available_agents": available_agents,
            "available_count": available_count,
            "supporting_agents": supporting_agents,
            "support_count": support_count,
            "agreement": agreement,
            "conflict": conflict,
            "conflict_state": candidate.get("conflict_state", "UNKNOWN"),
            "evidence_pattern": candidate.get("evidence_pattern", "UNKNOWN"),
            "evidence_strength": evidence_strength,
            "coverage_uncertainty": coverage_uncertainty,
            "evidence_uncertainty": evidence_uncertainty,
            "conflict_uncertainty": conflict_uncertainty,
            "uncertainty": overall_uncertainty,
            "confidence": confidence,
            "reliability": reliability,
        }

    def analyze(self, conflict_result):
        """Analyze either a result dictionary or a direct list of candidates."""
        if isinstance(conflict_result, dict):
            candidates = conflict_result.get("candidates", [])
        elif isinstance(conflict_result, list):
            candidates = conflict_result
        else:
            candidates = []
        return [self._analyze_candidate(candidate) for candidate in candidates]

    # ------------------------------------------------------------------
    # Visualization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _draw_text(image, text, position, scale=0.48, color=(225, 225, 225),
                   thickness=1):
        """Draw a consistently styled line of text on a panel."""
        cv2.putText(
            image,
            str(text),
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    def _draw_candidate_boxes(self, image, results):
        """Draw candidate boxes colored by their reliability label."""
        image_height, image_width = image.shape[:2]

        for index, candidate in enumerate(results, start=1):
            bbox = candidate.get("bbox", [0, 0, 0, 0])
            x, y, width, height = [
                int(round(float(value))) for value in bbox
            ]

            # Retain the original coordinate clamping behavior.
            x = max(0, min(x, image_width - 1))
            y = max(0, min(y, image_height - 1))
            width = max(1, min(width, image_width - x))
            height = max(1, min(height, image_height - y))

            reliability = candidate.get("reliability", "LOW")
            color = self.RELIABILITY_COLORS.get(
                reliability, self.RELIABILITY_COLORS["LOW"]
            )

            cv2.rectangle(
                image,
                (x, y),
                (x + width, y + height),
                color,
                2,
            )
            self._draw_text(
                image,
                f"C{index}",
                (x, max(y - 8, 20)),
                scale=0.55,
                color=color,
                thickness=2,
            )

    @staticmethod
    def _mean_field(results, field):
        """Return the mean of a result field, or zero for no candidates."""
        values = [float(item.get(field, 0.0)) for item in results]
        return float(np.mean(values)) if values else 0.0

    @staticmethod
    def _reliability_counts(results):
        """Count candidates assigned to each known reliability category."""
        counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for candidate in results:
            reliability = candidate.get("reliability", "LOW")
            if reliability in counts:
                counts[reliability] += 1
        return counts

    def _build_statistics_panel(self, image_height, results):
        """Create a compact panel with aggregate uncertainty statistics."""
        panel_width = 430
        panel = np.zeros((image_height, panel_width, 3), dtype=np.uint8)
        counts = self._reliability_counts(results)

        # A helper keeps each panel label concise and visually consistent.
        def line(text, y, scale=0.46, strong=False):
            self._draw_text(
                panel,
                text,
                (20, y),
                scale=scale,
                color=(255, 255, 255) if strong else (220, 220, 220),
                thickness=2 if strong else 1,
            )

        line("UNCERTAINTY AGENT", 35, 0.70, True)
        line(f"Candidates: {len(results)}", 70)

        line("UNCERTAINTY", 115, 0.58, True)
        line(f"Average: {self._mean_field(results, 'uncertainty'):.3f}", 150)
        line(f"Confidence: {self._mean_field(results, 'confidence'):.3f}", 180)

        line("RELIABILITY", 230, 0.58, True)
        line(f"High: {counts['HIGH']}", 265)
        line(f"Medium: {counts['MEDIUM']}", 295)
        line(f"Low: {counts['LOW']}", 325)

        line("UNCERTAINTY COMPONENTS", 370, 0.52, True)
        line(
            f"Coverage: {self._mean_field(results, 'coverage_uncertainty'):.3f}",
            405,
            0.42,
        )
        line(
            f"Evidence weakness: {self._mean_field(results, 'evidence_uncertainty'):.3f}",
            433,
            0.42,
        )
        line(
            f"Conflict: {self._mean_field(results, 'conflict_uncertainty'):.3f}",
            461,
            0.42,
        )
        return panel

    def save_visualization(
        self,
        rgb_path,
        results,
        output_path="outputs/uncertainty_analysis.png",
    ):
        """Save the RGB image with reliability boxes and a statistics panel.

        The panel layout is intentionally simplified compared with the original.
        This method does not alter candidate scores or analysis results.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        image = cv2.imread(str(rgb_path))
        if image is None:
            raise ValueError(f"Could not load RGB image: {rgb_path}")

        display_image = image.copy()
        self._draw_candidate_boxes(display_image, results)
        panel = self._build_statistics_panel(display_image.shape[0], results)
        final_image = np.hstack((display_image, panel))

        if not cv2.imwrite(str(output_path), final_image):
            raise OSError(f"Could not save visualization: {output_path}")

        print(f"Visualization saved to: {output_path}")
        return output_path


if __name__ == "__main__":
    agent = UncertaintyAgent()
    test_candidates = [
        {
            "bbox": [100, 100, 80, 80],
            "class": "test",
            "spatial_status": "DETECTED",
            "spectral_status": "DETECTED",
            "feature_status": "DETECTED",
            "spatial_evidence": 0.90,
            "spectral_evidence": 0.85,
            "feature_evidence": 0.80,
            "available_agents": ["spatial", "spectral", "feature"],
            "available_count": 3,
            "supporting_agents": ["spatial", "spectral", "feature"],
            "support_count": 3,
            "agreement": 1.0,
            "conflict": 0.10,
            "conflict_state": "LOW",
            "evidence_pattern": "THREE-AGENT AGREEMENT",
        }
    ]

    for index, result in enumerate(agent.analyze(test_candidates), start=1):
        print(f"\nCandidate {index}")
        for key in (
            "evidence_strength",
            "coverage_uncertainty",
            "evidence_uncertainty",
            "conflict_uncertainty",
            "uncertainty",
            "confidence",
            "reliability",
        ):
            value = result[key]
            label = key.replace("_", " ").title()
            print(f"  {label}: {value:.3f}" if isinstance(value, float) else f"  {label}: {value}")
