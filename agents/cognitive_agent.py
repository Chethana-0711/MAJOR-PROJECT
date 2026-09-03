class CognitiveAgent:

    def __init__(self):
        pass

    # --------------------------------------------------
    # COGNITIVE REASONING
    # --------------------------------------------------

    def reason(self, candidate):

        spatial_present = candidate.get(
            "spatial_present", False
        )

        spectral_present = candidate.get(
            "spectral_present", False
        )

        feature_present = candidate.get(
            "feature_present", False
        )

        spatial_confidence = candidate.get(
            "spatial_confidence", 0.0
        )

        thermal_evidence = candidate.get(
            "thermal_evidence", 0.0
        )

        feature_evidence = candidate.get(
            "feature_evidence", 0.0
        )

        uncertainty = candidate.get(
            "uncertainty", 1.0
        )

        conflict = candidate.get(
            "conflict", 1.0
        )

        agent_count = candidate.get(
            "agent_count", 0
        )

        # --------------------------------------------------
        # Evidence
        # --------------------------------------------------

        evidence = {
            "spatial": (
                spatial_confidence
                if spatial_present else 0.0
            ),

            "thermal": (
                thermal_evidence
                if spectral_present else 0.0
            ),

            "feature": (
                feature_evidence
                if feature_present else 0.0
            )
        }

        active_evidence = [
            value
            for value in evidence.values()
            if value > 0
        ]

        if active_evidence:

            evidence_strength = (
                sum(active_evidence) /
                len(active_evidence)
            )

        else:

            evidence_strength = 0.0

        strongest_agent = max(
            evidence,
            key=evidence.get
        )

        strongest_evidence = evidence[
            strongest_agent
        ]

        # --------------------------------------------------
        # Count strong agents
        # --------------------------------------------------

        strong_agents = sum(
            value >= 0.60
            for value in evidence.values()
        )

        weak_agents = sum(
            0 < value < 0.60
            for value in evidence.values()
        )

        # --------------------------------------------------
        # Detect evidence pattern
        # --------------------------------------------------

        pattern = "UNDEFINED"

        # All three strongly support
        if (
            spatial_present
            and spectral_present
            and feature_present
            and
            spatial_confidence >= 0.60
            and thermal_evidence >= 0.60
            and feature_evidence >= 0.60
        ):

            pattern = "TRIPLE_STRONG_SUPPORT"

        # Feature + thermal support, spatial missing/weak
        elif (
            spectral_present
            and feature_present
            and
            thermal_evidence >= 0.60
            and feature_evidence >= 0.60
            and
            (
                not spatial_present
                or spatial_confidence < 0.60
            )
        ):

            pattern = "VISUAL_DETECTOR_CONFLICT"

        # Spatial + thermal support, feature missing/weak
        elif (
            spatial_present
            and spectral_present
            and
            spatial_confidence >= 0.60
            and thermal_evidence >= 0.60
            and
            (
                not feature_present
                or feature_evidence < 0.60
            )
        ):

            pattern = "FEATURE_AGENT_CONFLICT"

        # Spatial + feature support, thermal missing/weak
        elif (
            spatial_present
            and feature_present
            and
            spatial_confidence >= 0.60
            and feature_evidence >= 0.60
            and
            (
                not spectral_present
                or thermal_evidence < 0.60
            )
        ):

            pattern = "THERMAL_AGENT_CONFLICT"

        # Strong spatial evidence but weak independent support
        elif (
            spatial_present
            and spatial_confidence >= 0.70
            and strong_agents == 1
        ):

            pattern = "UNSUPPORTED_SPATIAL_DETECTION"

        # Thermal-only anomaly
        elif (
            spectral_present
            and not spatial_present
            and not feature_present
        ):

            pattern = "THERMAL_ONLY"

        # Feature-only anomaly
        elif (
            feature_present
            and not spatial_present
            and not spectral_present
        ):

            pattern = "FEATURE_ONLY"

        # Spatial-only weak detection
        elif (
            spatial_present
            and not spectral_present
            and not feature_present
        ):

            pattern = "WEAK_SPATIAL_ONLY"

        # --------------------------------------------------
        # Interpret pattern
        # --------------------------------------------------

        meaningful_conflict = False

        # ----------------------------------------------
        # Triple agreement
        # ----------------------------------------------

        if pattern == "TRIPLE_STRONG_SUPPORT":

            decision = (
                "HIGH CONFIDENCE TARGET"
            )

            interpretation = (
                "Spatial, thermal, and feature "
                "agents independently provide strong "
                "support for the same candidate. "
                "The multimodal evidence is highly "
                "consistent."
            )

        # ----------------------------------------------
        # Visual detector conflict
        # ----------------------------------------------

        elif pattern == "VISUAL_DETECTOR_CONFLICT":

            meaningful_conflict = True

            decision = (
                "POTENTIAL CAMOUFLAGED TARGET"
            )

            interpretation = (
                "Thermal and feature agents provide "
                "strong evidence for the candidate "
                "while the spatial detector is absent "
                "or weak. The disagreement is retained "
                "as meaningful evidence because a "
                "camouflaged target may be difficult "
                "for the spatial detector to recognize."
            )

        # ----------------------------------------------
        # Feature conflict
        # ----------------------------------------------

        elif pattern == "FEATURE_AGENT_CONFLICT":

            meaningful_conflict = True

            decision = (
                "LIKELY TARGET"
            )

            interpretation = (
                "Spatial and thermal agents provide "
                "strong evidence while the feature "
                "agent provides weak or absent support. "
                "The disagreement is retained rather "
                "than automatically discarding the "
                "candidate."
            )

        # ----------------------------------------------
        # Thermal conflict
        # ----------------------------------------------

        elif pattern == "THERMAL_AGENT_CONFLICT":

            meaningful_conflict = True

            decision = (
                "LIKELY TARGET"
            )

            interpretation = (
                "Spatial and feature agents provide "
                "strong evidence while thermal evidence "
                "is weak or absent. This may indicate "
                "that the target has limited thermal "
                "contrast with its background. The "
                "thermal disagreement is therefore "
                "retained for interpretation."
            )

        # ----------------------------------------------
        # Unsupported spatial detection
        # ----------------------------------------------

        elif pattern == "UNSUPPORTED_SPATIAL_DETECTION":

            decision = (
                "UNSUPPORTED SPATIAL DETECTION"
            )

            interpretation = (
                "The spatial detector reports a strong "
                "candidate, but the independent agents "
                "do not provide sufficient support. "
                "The candidate is retained but treated "
                "with caution."
            )

        # ----------------------------------------------
        # Thermal only
        # ----------------------------------------------

        elif pattern == "THERMAL_ONLY":

            decision = (
                "THERMAL ANOMALY"
            )

            interpretation = (
                "The thermal agent identifies an "
                "anomalous region without independent "
                "spatial or feature support. This "
                "evidence is retained but is not "
                "sufficient by itself to establish "
                "a target."
            )

        # ----------------------------------------------
        # Feature only
        # ----------------------------------------------

        elif pattern == "FEATURE_ONLY":

            decision = (
                "FEATURE ANOMALY"
            )

            interpretation = (
                "The feature agent identifies an "
                "anomalous region without independent "
                "spatial or thermal support. The "
                "candidate is retained for further "
                "reasoning."
            )

        # ----------------------------------------------
        # Weak spatial only
        # ----------------------------------------------

        elif pattern == "WEAK_SPATIAL_ONLY":

            decision = (
                "LIKELY FALSE POSITIVE"
            )

            interpretation = (
                "The spatial detector provides weak "
                "evidence and neither thermal nor "
                "feature agents independently support "
                "the candidate."
            )

        # ----------------------------------------------
        # Fallback
        # ----------------------------------------------

        else:

            decision = (
                "UNCERTAIN CANDIDATE"
            )

            interpretation = (
                "The available evidence does not form "
                "a sufficiently clear multimodal pattern."
            )

        # --------------------------------------------------
        # Conflict severity
        # --------------------------------------------------

        # We distinguish:
        #
        # conflict = numerical disagreement
        #
        # meaningful conflict =
        # disagreement that contains a useful pattern
        #

        if meaningful_conflict:

            conflict_interpretation = (
                "Conflict is considered meaningful "
                "because strong evidence from multiple "
                "agents is inconsistent with another "
                "agent."
            )

        elif conflict >= 0.50:

            conflict_interpretation = (
                "High numerical conflict is present, "
                "but the available evidence does not "
                "form a strong multimodal camouflage "
                "pattern."
            )

        else:

            conflict_interpretation = (
                "No strong meaningful conflict pattern "
                "was identified."
            )

        # --------------------------------------------------
        # Cognitive confidence
        # --------------------------------------------------

        # Evidence contribution
        evidence_confidence = (
            evidence_strength
        )

        # Multi-agent contribution
        agent_confidence = (
            agent_count / 3.0
        )

        # Uncertainty contribution
        uncertainty_confidence = (
            1.0 - uncertainty
        )

        confidence = (

            0.45 *
            evidence_confidence

            +

            0.30 *
            agent_confidence

            +

            0.25 *
            uncertainty_confidence
        )

        # Meaningful conflict should NOT automatically
        # destroy confidence.
        #
        # It means the evidence needs interpretation.

        if meaningful_conflict:

            confidence += 0.05

        confidence = max(
            0.0,
            min(
                1.0,
                confidence
            )
        )

        # --------------------------------------------------
        # Final explanation
        # --------------------------------------------------

        explanation = (
            interpretation
            + " "
            + conflict_interpretation
        )

        # --------------------------------------------------
        # Return result
        # --------------------------------------------------

        return {

            "decision":
                decision,

            "confidence":
                float(confidence),

            "meaningful_conflict":
                meaningful_conflict,

            "conflict_pattern":
                pattern,

            "strongest_agent":
                strongest_agent,

            "strongest_evidence":
                float(strongest_evidence),

            "evidence_strength":
                float(evidence_strength),

            "strong_agents":
                int(strong_agents),

            "weak_agents":
                int(weak_agents),

            "interpretation":
                explanation
        }

    # --------------------------------------------------
    # ANALYZE ALL CANDIDATES
    # --------------------------------------------------

    def analyze(
        self,
        uncertainty_results
    ):

        results = []

        for candidate in uncertainty_results:

            cognitive_result = (
                self.reason(candidate)
            )

            combined = {
                **candidate,
                **cognitive_result
            }

            results.append(
                combined
            )

        return results


# --------------------------------------------------
# TEST CASES
# --------------------------------------------------

if __name__ == "__main__":

    agent = CognitiveAgent()

    test_cases = [

        # ------------------------------------------
        # CASE 1
        # All agents strongly agree
        # ------------------------------------------

        {
            "name":
                "Triple agreement",

            "spatial_present":
                True,

            "spectral_present":
                True,

            "feature_present":
                True,

            "spatial_confidence":
                0.86,

            "thermal_evidence":
                0.79,

            "feature_evidence":
                0.88,

            "agent_count":
                3,

            "uncertainty":
                0.08,

            "conflict":
                0.05
        },

        # ------------------------------------------
        # CASE 2
        # YOLO misses, thermal + feature strong
        # ------------------------------------------

        {
            "name":
                "Meaningful camouflage conflict",

            "spatial_present":
                False,

            "spectral_present":
                True,

            "feature_present":
                True,

            "spatial_confidence":
                0.0,

            "thermal_evidence":
                0.82,

            "feature_evidence":
                0.88,

            "agent_count":
                2,

            "uncertainty":
                0.25,

            "conflict":
                0.70
        },

        # ------------------------------------------
        # CASE 3
        # YOLO only
        # ------------------------------------------

        {
            "name":
                "Unsupported spatial detection",

            "spatial_present":
                True,

            "spectral_present":
                False,

            "feature_present":
                False,

            "spatial_confidence":
                0.81,

            "thermal_evidence":
                0.0,

            "feature_evidence":
                0.0,

            "agent_count":
                1,

            "uncertainty":
                0.70,

            "conflict":
                1.0
        },

        # ------------------------------------------
        # CASE 4
        # Thermal only
        # ------------------------------------------

        {
            "name":
                "Thermal anomaly",

            "spatial_present":
                False,

            "spectral_present":
                True,

            "feature_present":
                False,

            "spatial_confidence":
                0.0,

            "thermal_evidence":
                0.70,

            "feature_evidence":
                0.0,

            "agent_count":
                1,

            "uncertainty":
                0.60,

            "conflict":
                1.0
        }
    ]

    print(
        "\nCOGNITIVE AGENT"
    )

    print(
        "-------------------------"
    )

    for case in test_cases:

        result = agent.reason(case)

        print(
            f"\nTest: "
            f"{case['name']}"
        )

        print(
            f"Decision: "
            f"{result['decision']}"
        )

        print(
            f"Conflict pattern: "
            f"{result['conflict_pattern']}"
        )

        print(
            f"Meaningful conflict: "
            f"{result['meaningful_conflict']}"
        )

        print(
            f"Confidence: "
            f"{result['confidence']:.3f}"
        )

        print(
            "Reasoning:"
        )

        print(
            result["interpretation"]
        )