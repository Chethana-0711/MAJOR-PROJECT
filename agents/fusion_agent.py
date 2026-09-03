import math


class FusionAgent:

    def __init__(self, overlap_threshold=0.10):
        self.overlap_threshold = overlap_threshold

    def calculate_iou(self, box1, box2):
        """
        Calculate IoU between two boxes.

        box format:
        [x, y, width, height]
        """

        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # Convert to x1, y1, x2, y2
        a_x2 = x1 + w1
        a_y2 = y1 + h1

        b_x2 = x2 + w2
        b_y2 = y2 + h2

        # Intersection
        ix1 = max(x1, x2)
        iy1 = max(y1, y2)
        ix2 = min(a_x2, b_x2)
        iy2 = min(a_y2, b_y2)

        intersection_width = max(0, ix2 - ix1)
        intersection_height = max(0, iy2 - iy1)

        intersection = (
            intersection_width *
            intersection_height
        )

        # Areas
        area1 = w1 * h1
        area2 = w2 * h2

        union = area1 + area2 - intersection

        if union == 0:
            return 0.0

        return intersection / union

    def fuse(self, spatial_predictions, spectral_regions):

        results = []

        for detection in spatial_predictions:

            spatial_box = detection["bbox"]
            spatial_confidence = detection["confidence"]
            class_name = detection["class"]

            best_thermal_evidence = 0.0
            best_overlap = 0.0

            # Compare this YOLO detection
            # against every thermal region.
            for region in spectral_regions:

                spectral_box = region["bbox"]

                overlap = self.calculate_iou(
                    spatial_box,
                    spectral_box
                )

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_thermal_evidence = (
                        region["evidence"]
                    )

            # --------------------------------------------------
            # Agreement
            # --------------------------------------------------

            if best_overlap >= self.overlap_threshold:

                agreement = (
                    spatial_confidence +
                    best_thermal_evidence
                ) / 2

            else:

                agreement = 0.0

            # --------------------------------------------------
            # Conflict
            # --------------------------------------------------

            conflict = (
                abs(
                    spatial_confidence -
                    best_thermal_evidence
                )
            )

            # If there is no spatial overlap,
            # treat it as stronger disagreement.
            if best_overlap < self.overlap_threshold:
                conflict = min(
                    1.0,
                    conflict + 0.25
                )

            # --------------------------------------------------
            # Uncertainty
            # --------------------------------------------------

            uncertainty = (
                1.0 -
                max(
                    spatial_confidence,
                    best_thermal_evidence
                )
            )

            # --------------------------------------------------
            # Suspicion score
            # --------------------------------------------------

            suspicion = (
                0.40 * spatial_confidence +
                0.30 * best_thermal_evidence +
                0.20 * conflict +
                0.10 * uncertainty
            )

            suspicion = max(
                0.0,
                min(1.0, suspicion)
            )

            # --------------------------------------------------
            # Final decision
            # --------------------------------------------------

            if suspicion >= 0.70:
                decision = "HIGH SUSPICION"

            elif suspicion >= 0.45:
                decision = "MODERATE SUSPICION"

            else:
                decision = "LOW SUSPICION"

            results.append({
                "class": class_name,
                "bbox": spatial_box,
                "spatial_confidence": spatial_confidence,
                "thermal_evidence": best_thermal_evidence,
                "overlap": best_overlap,
                "agreement": agreement,
                "conflict": conflict,
                "uncertainty": uncertainty,
                "suspicion_score": suspicion,
                "decision": decision
            })

        return results


# --------------------------------------------------
# TEST
# --------------------------------------------------

if __name__ == "__main__":

    agent = FusionAgent()

    # Example Spatial Agent output
    spatial = [
        {
            "class": "persongreen",
            "bbox": [100, 100, 50, 60],
            "confidence": 0.82
        }
    ]

    # Example Spectral Agent output
    spectral = [
        {
            "bbox": [105, 105, 45, 55],
            "evidence": 0.75
        }
    ]

    result = agent.fuse(
        spatial,
        spectral
    )

    print("\nFUSION AGENT")
    print("-------------------------")

    for item in result:

        print(
            f"\nClass: "
            f"{item['class']}"
        )

        print(
            f"Spatial confidence: "
            f"{item['spatial_confidence']:.3f}"
        )

        print(
            f"Thermal evidence: "
            f"{item['thermal_evidence']:.3f}"
        )

        print(
            f"Overlap: "
            f"{item['overlap']:.3f}"
        )

        print(
            f"Agreement: "
            f"{item['agreement']:.3f}"
        )

        print(
            f"Conflict: "
            f"{item['conflict']:.3f}"
        )

        print(
            f"Uncertainty: "
            f"{item['uncertainty']:.3f}"
        )

        print(
            f"Suspicion score: "
            f"{item['suspicion_score']:.3f}"
        )

        print(
            f"Decision: "
            f"{item['decision']}"
        )