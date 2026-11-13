from pathlib import Path

import cv2
import numpy as np


# ============================================================
# CONFLICT AGENT
# ============================================================

class ConflictAgent:

    def __init__(self):

        # Region matching thresholds
        self.iou_threshold = 0.10

        self.containment_threshold = 0.30

        self.center_similarity_threshold = 0.70

    # ========================================================
    # IOU
    # ========================================================

    def calculate_iou(self, box_a, box_b):

        ax, ay, aw, ah = box_a
        bx, by, bw, bh = box_b

        ax2 = ax + aw
        ay2 = ay + ah

        bx2 = bx + bw
        by2 = by + bh

        intersection_x1 = max(ax, bx)
        intersection_y1 = max(ay, by)

        intersection_x2 = min(ax2, bx2)
        intersection_y2 = min(ay2, by2)

        intersection_w = max(
            0,
            intersection_x2 - intersection_x1
        )

        intersection_h = max(
            0,
            intersection_y2 - intersection_y1
        )

        intersection_area = (
            intersection_w *
            intersection_h
        )

        area_a = aw * ah
        area_b = bw * bh

        union_area = (
            area_a +
            area_b -
            intersection_area
        )

        if union_area <= 0:
            return 0.0

        return intersection_area / union_area

    # ========================================================
    # CONTAINMENT
    # ========================================================

    def calculate_containment(
        self,
        box_a,
        box_b
    ):

        ax, ay, aw, ah = box_a
        bx, by, bw, bh = box_b

        ax2 = ax + aw
        ay2 = ay + ah

        bx2 = bx + bw
        by2 = by + bh

        intersection_x1 = max(ax, bx)
        intersection_y1 = max(ay, by)

        intersection_x2 = min(ax2, bx2)
        intersection_y2 = min(ay2, by2)

        intersection_w = max(
            0,
            intersection_x2 - intersection_x1
        )

        intersection_h = max(
            0,
            intersection_y2 - intersection_y1
        )

        intersection_area = (
            intersection_w *
            intersection_h
        )

        smaller_area = min(
            aw * ah,
            bw * bh
        )

        if smaller_area <= 0:
            return 0.0

        return (
            intersection_area /
            smaller_area
        )

    # ========================================================
    # CENTER SIMILARITY
    # ========================================================

    def calculate_center_similarity(
        self,
        box_a,
        box_b
    ):

        ax, ay, aw, ah = box_a
        bx, by, bw, bh = box_b

        center_a = (
            ax + aw / 2,
            ay + ah / 2
        )

        center_b = (
            bx + bw / 2,
            by + bh / 2
        )

        distance = np.sqrt(
            (
                center_a[0] -
                center_b[0]
            ) ** 2
            +
            (
                center_a[1] -
                center_b[1]
            ) ** 2
        )

        diagonal_a = np.sqrt(
            aw ** 2 +
            ah ** 2
        )

        diagonal_b = np.sqrt(
            bw ** 2 +
            bh ** 2
        )

        reference_distance = max(
            diagonal_a,
            diagonal_b,
            1.0
        )

        normalized_distance = (
            distance /
            reference_distance
        )

        similarity = 1.0 - min(
            normalized_distance,
            1.0
        )

        return float(similarity)

    # ========================================================
    # REGION MATCHING
    # ========================================================

    def regions_match(
        self,
        box_a,
        box_b
    ):

        iou = self.calculate_iou(
            box_a,
            box_b
        )

        containment = (
            self.calculate_containment(
                box_a,
                box_b
            )
        )

        center_similarity = (
            self.calculate_center_similarity(
                box_a,
                box_b
            )
        )

        matched = (
            iou >= self.iou_threshold
            or
            containment >= self.containment_threshold
            or
            center_similarity >=
            self.center_similarity_threshold
        )

        # A single score is useful when several possible
        # matches exist. We choose the strongest geometric
        # relationship instead of simply taking the first one.

        match_score = max(
            iou,
            containment,
            center_similarity
        )

        return {

            "matched":
                matched,

            "iou":
                float(iou),

            "containment":
                float(containment),

            "center_similarity":
                float(center_similarity),

            "match_score":
                float(match_score)
        }

    # ========================================================
    # UNION OF TWO BOXES
    # ========================================================

    def union_boxes(
        self,
        box_a,
        box_b
    ):

        ax, ay, aw, ah = box_a
        bx, by, bw, bh = box_b

        x1 = min(
            ax,
            bx
        )

        y1 = min(
            ay,
            by
        )

        x2 = max(
            ax + aw,
            bx + bw
        )

        y2 = max(
            ay + ah,
            by + bh
        )

        return [
            int(x1),
            int(y1),
            int(x2 - x1),
            int(y2 - y1)
        ]

    # ========================================================
    # GET SPATIAL REGIONS
    # ========================================================

    def _get_spatial_regions(
        self,
        spatial_result
    ):

        if spatial_result is None:
            return []

        if isinstance(
            spatial_result,
            list
        ):

            return spatial_result

        if isinstance(
            spatial_result,
            dict
        ):

            return spatial_result.get(
                "detections",
                []
            )

        return []

    # ========================================================
    # GET SPECTRAL REGIONS
    # ========================================================

    def _get_spectral_regions(
        self,
        spectral_result
    ):

        if spectral_result is None:
            return []

        if isinstance(
            spectral_result,
            list
        ):

            return spectral_result

        if isinstance(
            spectral_result,
            dict
        ):

            return spectral_result.get(
                "regions",
                []
            )

        return []

    # ========================================================
    # GET FEATURE REGIONS
    # ========================================================

    def _get_feature_regions(
        self,
        result
    ):

        if result is None:
            return []

        if isinstance(
            result,
            dict
        ):

            regions = list(
                result.get(
                    "regions",
                    []
                )
            )

            localized_regions = (
                result.get(
                    "localized_evidence_regions",
                    []
                )
            )

            regions.extend(
                localized_regions
            )

            return regions

        return result

    # ========================================================
    # SPATIAL EVIDENCE
    # ========================================================

    def _spatial_evidence(
        self,
        detection
    ):

        return float(
            detection.get(
                "confidence",
                0.0
            )
        )

    # ========================================================
    # SPECTRAL EVIDENCE
    # ========================================================

    def _spectral_evidence(
        self,
        region
    ):

        if "combined_evidence" in region:

            return float(
                region[
                    "combined_evidence"
                ]
            )

        if "evidence" in region:

            return float(
                region[
                    "evidence"
                ]
            )

        if "ml_probability" in region:

            return float(
                region[
                    "ml_probability"
                ]
            )

        return 0.0

    # ========================================================
    # FEATURE EVIDENCE
    # ========================================================

    def _feature_evidence(
        self,
        region
    ):

        return float(
            region.get(
                "evidence",
                0.0
            )
        )

    # ========================================================
    # BUILD CANDIDATES
    #
    # IMPORTANT:
    #
    # The previous implementation used a greedy:
    #
    #     first matching region + break
    #
    # This could cause a legitimate Spectral region to be
    # associated with the wrong Spatial candidate or disappear
    # during later duplicate checks.
    #
    # This implementation:
    #
    #     1. Starts with ALL Spatial regions.
    #     2. Finds the BEST Spatial-Spectral matches.
    #     3. Keeps unmatched Spectral regions.
    #     4. Finds the BEST candidate-Feature matches.
    #     5. Keeps unmatched Feature regions.
    #
    # Therefore every agent region is preserved.
    # ========================================================

    def _build_candidates(
        self,
        spatial_regions,
        spectral_regions,
        feature_regions
    ):

        candidates = []

        # ====================================================
        # STEP 1
        # CREATE A CANDIDATE FOR EVERY SPATIAL REGION
        # ====================================================

        for spatial in spatial_regions:

            candidates.append({

                "bbox":
                    spatial["bbox"],

                "class":
                    spatial.get(
                        "class",
                        "unknown"
                    ),

                "spatial":
                    spatial,

                "spectral":
                    None,

                "feature":
                    None
            })

        # ====================================================
        # STEP 2
        # GLOBAL SPATIAL ↔ SPECTRAL MATCHING
        #
        # Instead of taking the first matching spectral region,
        # calculate ALL valid matches and select the strongest
        # ones first.
        # ====================================================

        possible_matches = []

        for spatial_index, candidate in enumerate(
            candidates
        ):

            spatial = candidate[
                "spatial"
            ]

            if spatial is None:
                continue

            for spectral_index, spectral in enumerate(
                spectral_regions
            ):

                match = self.regions_match(
                    spatial["bbox"],
                    spectral["bbox"]
                )

                if not match["matched"]:
                    continue

                possible_matches.append({

                    "spatial_index":
                        spatial_index,

                    "spectral_index":
                        spectral_index,

                    "score":
                        match["match_score"],

                    "iou":
                        match["iou"],

                    "containment":
                        match["containment"],

                    "center_similarity":
                        match[
                            "center_similarity"
                        ]
                })

        # Strongest matches first

        possible_matches.sort(
            key=lambda item:
                item["score"],
            reverse=True
        )

        used_spatial = set()

        used_spectral = set()

        for match in possible_matches:

            spatial_index = match[
                "spatial_index"
            ]

            spectral_index = match[
                "spectral_index"
            ]

            # One spectral region should correspond to one
            # candidate, and one spatial detection should not
            # receive several unrelated spectral regions.

            if spatial_index in used_spatial:
                continue

            if spectral_index in used_spectral:
                continue

            candidate = candidates[
                spatial_index
            ]

            spectral = spectral_regions[
                spectral_index
            ]

            candidate[
                "spectral"
            ] = spectral

            # Expand displayed candidate box so that both
            # pieces of evidence are represented.

            candidate[
                "bbox"
            ] = self.union_boxes(
                candidate["bbox"],
                spectral["bbox"]
            )

            used_spatial.add(
                spatial_index
            )

            used_spectral.add(
                spectral_index
            )

        # ====================================================
        # STEP 3
        # EVERY UNMATCHED SPECTRAL REGION BECOMES ITS OWN
        # CANDIDATE
        # ====================================================

        for spectral_index, spectral in enumerate(
            spectral_regions
        ):

            if spectral_index in used_spectral:
                continue

            candidates.append({

                "bbox":
                    spectral["bbox"],

                "class":
                    "unknown",

                "spatial":
                    None,

                "spectral":
                    spectral,

                "feature":
                    None
            })

        # ====================================================
        # STEP 4
        # FEATURE MATCHING
        #
        # Feature evidence is matched against the current
        # candidate groups.
        # ====================================================

        feature_matches = []

        for feature_index, feature in enumerate(
            feature_regions
        ):

            for candidate_index, candidate in enumerate(
                candidates
            ):

                # Do not attach a Feature region to a candidate
                # that already contains Feature evidence.

                if candidate[
                    "feature"
                ] is not None:

                    continue

                match = self.regions_match(
                    candidate["bbox"],
                    feature["bbox"]
                )

                if not match["matched"]:
                    continue

                feature_matches.append({

                    "feature_index":
                        feature_index,

                    "candidate_index":
                        candidate_index,

                    "score":
                        match["match_score"],

                    "iou":
                        match["iou"],

                    "containment":
                        match["containment"],

                    "center_similarity":
                        match[
                            "center_similarity"
                        ]
                })

        # Strongest matches first

        feature_matches.sort(
            key=lambda item:
                item["score"],
            reverse=True
        )

        used_features = set()

        used_feature_candidates = set()

        for match in feature_matches:

            feature_index = match[
                "feature_index"
            ]

            candidate_index = match[
                "candidate_index"
            ]

            if feature_index in used_features:
                continue

            if (
                candidate_index
                in used_feature_candidates
            ):
                continue

            candidate = candidates[
                candidate_index
            ]

            feature = feature_regions[
                feature_index
            ]

            candidate[
                "feature"
            ] = feature

            candidate[
                "bbox"
            ] = self.union_boxes(
                candidate["bbox"],
                feature["bbox"]
            )

            used_features.add(
                feature_index
            )

            used_feature_candidates.add(
                candidate_index
            )

        # ====================================================
        # STEP 5
        # EVERY UNMATCHED FEATURE REGION BECOMES A CANDIDATE
        # ====================================================

        for feature_index, feature in enumerate(
            feature_regions
        ):

            if feature_index in used_features:
                continue

            candidates.append({

                "bbox":
                    feature["bbox"],

                "class":
                    "unknown",

                "spatial":
                    None,

                "spectral":
                    None,

                "feature":
                    feature
            })

        return candidates

    # ========================================================
    # ANALYZE ONE CANDIDATE
    # ========================================================

    def _analyze_candidate(
        self,
        candidate,
        spatial_available,
        spectral_available,
        feature_available
    ):

        spatial = candidate[
            "spatial"
        ]

        spectral = candidate[
            "spectral"
        ]

        feature = candidate[
            "feature"
        ]

        # ====================================================
        # STATUS
        # ====================================================

        if spatial is not None:

            spatial_status = (
                "DETECTED"
            )

        elif spatial_available:

            spatial_status = (
                "NO_DETECTION"
            )

        else:

            spatial_status = (
                "NOT_AVAILABLE"
            )

        if spectral is not None:

            spectral_status = (
                "DETECTED"
            )

        elif spectral_available:

            spectral_status = (
                "NO_DETECTION"
            )

        else:

            spectral_status = (
                "NOT_AVAILABLE"
            )

        if feature is not None:

            if (
                feature.get(
                    "evidence_type",
                    "detection"
                )
                ==
                "localized"
            ):

                feature_status = (
                    "EVIDENCE_PRESENT"
                )

            else:

                feature_status = (
                    "DETECTED"
                )

        elif feature_available:

            feature_status = (
                "NO_DETECTION"
            )

        else:

            feature_status = (
                "NOT_AVAILABLE"
            )

        # ====================================================
        # EVIDENCE
        # ====================================================

        spatial_evidence = (

            self._spatial_evidence(
                spatial
            )

            if spatial is not None

            else 0.0
        )

        spectral_evidence = (

            self._spectral_evidence(
                spectral
            )

            if spectral is not None

            else 0.0
        )

        feature_evidence = (

            self._feature_evidence(
                feature
            )

            if feature is not None

            else 0.0
        )

        # ====================================================
        # AVAILABLE AGENTS
        # ====================================================

        available_agents = []

        if spatial_available:

            available_agents.append(
                "spatial"
            )

        if spectral_available:

            available_agents.append(
                "spectral"
            )

        if feature_available:

            available_agents.append(
                "feature"
            )

        available_count = len(
            available_agents
        )

        # ====================================================
        # SUPPORTING AGENTS
        # ====================================================

        supporting_agents = []

        if spatial is not None:

            supporting_agents.append(
                "spatial"
            )

        if spectral is not None:

            supporting_agents.append(
                "spectral"
            )

        if feature is not None:

            feature_type = feature.get(
                "evidence_type",
                "detection"
            )

            # Localized feature evidence is preserved, but
            # does not count as full feature detection.

            if feature_type != "localized":

                supporting_agents.append(
                    "feature"
                )

        support_count = len(
            supporting_agents
        )

        # ====================================================
        # NON-DETECTING AGENTS
        # ====================================================

        non_detecting_agents = [

            agent

            for agent in available_agents

            if agent
            not in supporting_agents
        ]

        # ====================================================
        # MISSING AGENTS
        # ====================================================

        missing_agents = [

            agent

            for agent in [
                "spatial",
                "spectral",
                "feature"
            ]

            if agent
            not in available_agents
        ]

        # ====================================================
        # AGREEMENT
        # ====================================================

        if available_count > 0:

            agreement = (
                support_count /
                available_count
            )

        else:

            agreement = 0.0

        # ====================================================
        # EVIDENCE VALUES
        #
        # NOT_AVAILABLE is excluded.
        # NO_DETECTION contributes zero.
        # ====================================================

        evidence_values = []

        if spatial_available:

            evidence_values.append(
                spatial_evidence
            )

        if spectral_available:

            evidence_values.append(
                spectral_evidence
            )

        if feature_available:

            evidence_values.append(
                feature_evidence
            )

        # ====================================================
        # CONFLICT
        # ====================================================

        if len(
            evidence_values
        ) >= 2:

            evidence_range = (
                max(
                    evidence_values
                )
                -
                min(
                    evidence_values
                )
            )

            conflict = float(
                np.clip(
                    evidence_range,
                    0.0,
                    1.0
                )
            )

        else:

            conflict = 0.0

        # ====================================================
        # PATTERN
        # ====================================================

        if support_count == 0:

            pattern = (
                "NO DETECTION"
            )

        elif (
            support_count ==
            available_count
        ):

            if available_count == 3:

                pattern = (
                    "TRIPLE AGREEMENT"
                )

            else:

                pattern = (
                    "FULL AVAILABLE AGREEMENT"
                )

        elif supporting_agents == [
            "spatial"
        ]:

            pattern = (
                "SPATIAL-ONLY EVIDENCE"
            )

        elif supporting_agents == [
            "spectral"
        ]:

            pattern = (
                "THERMAL-ONLY EVIDENCE"
            )

        elif supporting_agents == [
            "feature"
        ]:

            pattern = (
                "FEATURE-ONLY EVIDENCE"
            )

        else:

            pattern = (
                "PARTIAL AGREEMENT"
            )

        # ====================================================
        # CONFLICT STATE
        # ====================================================

        if available_count < 2:

            conflict_state = (
                "UNKNOWN"
            )

        elif conflict < 0.20:

            conflict_state = (
                "LOW"
            )

        elif conflict < 0.50:

            conflict_state = (
                "MODERATE"
            )

        else:

            conflict_state = (
                "HIGH"
            )

        # ====================================================
        # AVERAGE EVIDENCE
        # ====================================================

        if evidence_values:

            average_evidence = float(
                np.mean(
                    evidence_values
                )
            )

        else:

            average_evidence = 0.0

        # ====================================================
        # RESULT
        # ====================================================

        return {

            "bbox":
                candidate["bbox"],

            "class":
                candidate["class"],

            # Status
            "spatial_status":
                spatial_status,

            "spectral_status":
                spectral_status,

            "feature_status":
                feature_status,

            # Evidence
            "spatial_evidence":
                spatial_evidence,

            "spectral_evidence":
                spectral_evidence,

            "feature_evidence":
                feature_evidence,

            # Availability
            "available_agents":
                available_agents,

            "available_count":
                available_count,

            "missing_agents":
                missing_agents,

            # Detection
            "supporting_agents":
                supporting_agents,

            "support_count":
                support_count,

            "non_detecting_agents":
                non_detecting_agents,

            # Analysis
            "agreement":
                float(
                    agreement
                ),

            "conflict":
                conflict,

            "conflict_state":
                conflict_state,

            "average_evidence":
                average_evidence,

            "evidence_pattern":
                pattern
        }

    # ========================================================
    # MAIN ANALYSIS
    # ========================================================

    def analyze(
        self,
        spatial_result=None,
        spectral_result=None,
        feature_result=None
    ):

        # ----------------------------------------------------
        # Determine available agents
        # ----------------------------------------------------

        spatial_available = (
            spatial_result is not None
        )

        spectral_available = (
            spectral_result is not None
        )

        feature_available = (
            feature_result is not None
        )

        # ----------------------------------------------------
        # Extract regions
        # ----------------------------------------------------

        spatial_regions = (
            self._get_spatial_regions(
                spatial_result
            )
        )

        spectral_regions = (
            self._get_spectral_regions(
                spectral_result
            )
        )

        feature_regions = (
            self._get_feature_regions(
                feature_result
            )
        )

        # ----------------------------------------------------
        # Build candidates
        # ----------------------------------------------------

        candidates = (
            self._build_candidates(
                spatial_regions,
                spectral_regions,
                feature_regions
            )
        )

        results = []

        for candidate in candidates:

            candidate_result = (
                self._analyze_candidate(
                    candidate,
                    spatial_available,
                    spectral_available,
                    feature_available
                )
            )

            results.append(
                candidate_result
            )

        # ----------------------------------------------------
        # Available agents
        # ----------------------------------------------------

        available_agents = []

        if spatial_available:

            available_agents.append(
                "spatial"
            )

        if spectral_available:

            available_agents.append(
                "spectral"
            )

        if feature_available:

            available_agents.append(
                "feature"
            )

        # ====================================================
        # DETAILED TERMINAL OUTPUT
        # ====================================================

        print()
        print(
            "Detailed Conflict Analysis"
        )
        print(
            "=" * 70
        )

        for index, candidate in enumerate(
            results,
            start=1
        ):

            print()
            print(
                f"Candidate {index}"
            )

            print(
                f"  Class: "
                f"{candidate.get('class', 'unknown')}"
            )

            print(
                f"  BBox: "
                f"{candidate.get('bbox', [])}"
            )

            print(
                f"  Spatial: "
                f"{candidate.get('spatial_status', 'UNKNOWN')} "
                f"{candidate.get('spatial_evidence', 0.0):.3f}"
            )

            print(
                f"  Thermal: "
                f"{candidate.get('spectral_status', 'UNKNOWN')} "
                f"{candidate.get('spectral_evidence', 0.0):.3f}"
            )

            print(
                f"  Feature: "
                f"{candidate.get('feature_status', 'UNKNOWN')} "
                f"{candidate.get('feature_evidence', 0.0):.3f}"
            )

            print(
                f"  Available: "
                f"{candidate.get('available_count', 0)}/3"
            )

            print(
                f"  Supporting: "
                f"{candidate.get('support_count', 0)}/"
                f"{candidate.get('available_count', 0)}"
            )

            print(
                f"  Supporting agents: "
                f"{candidate.get('supporting_agents', [])}"
            )

            print(
                f"  Pattern: "
                f"{candidate.get('evidence_pattern', 'UNKNOWN')}"
            )

            print(
                f"  Agreement: "
                f"{candidate.get('agreement', 0.0):.3f}"
            )

            print(
                f"  Conflict: "
                f"{candidate.get('conflict', 0.0):.3f} "
                f"({candidate.get('conflict_state', 'UNKNOWN')})"
            )

            print(
                f"  Average evidence: "
                f"{candidate.get('average_evidence', 0.0):.3f}"
            )

        return {

            "agent":
                "conflict",

            "available_agents":
                available_agents,

            "available_agent_count":
                len(
                    available_agents
                ),

            "candidates":
                results,

            "candidate_count":
                len(
                    results
                )
        }

    # ========================================================
    # VISUALIZATION
    # ========================================================

    def save_visualization(
        self,
        rgb_path,
        results,
        output_path="outputs/conflict_analysis.png"
    ):

        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        image = cv2.imread(
            str(rgb_path)
        )

        if image is None:

            raise ValueError(
                f"Could not load image: "
                f"{rgb_path}"
            )

        candidates = results.get(
            "candidates",
            []
        )

        # ====================================================
        # EVIDENCE AGENTS
        # ====================================================

        def get_evidence_agents(
            candidate
        ):

            agents = []

            if candidate.get(
                "spatial_status"
            ) == "DETECTED":

                agents.append(
                    "Spatial"
                )

            if candidate.get(
                "spectral_status"
            ) == "DETECTED":

                agents.append(
                    "Thermal"
                )

            if candidate.get(
                "feature_status"
            ) in [
                "DETECTED",
                "EVIDENCE_PRESENT"
            ]:

                agents.append(
                    "Feature"
                )

            return agents

        # ====================================================
        # DRAW CANDIDATES
        # ====================================================

        for index, candidate in enumerate(
            candidates,
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

            # Keep coordinates inside image

            x = max(
                0,
                min(
                    x,
                    image.shape[1] - 1
                )
            )

            y = max(
                0,
                min(
                    y,
                    image.shape[0] - 1
                )
            )

            w = max(
                1,
                min(
                    w,
                    image.shape[1] - x
                )
            )

            h = max(
                1,
                min(
                    h,
                    image.shape[0] - y
                )
            )

            evidence_agents = (
                get_evidence_agents(
                    candidate
                )
            )

            evidence_count = len(
                evidence_agents
            )

            # ------------------------------------------------
            # GREEN = 3 AGENTS
            # YELLOW = 2 AGENTS
            # RED = 1 AGENT
            # ------------------------------------------------

            if evidence_count >= 3:

                color = (
                    0,
                    200,
                    0
                )

            elif evidence_count == 2:

                color = (
                    0,
                    220,
                    220
                )

            elif evidence_count == 1:

                color = (
                    0,
                    0,
                    255
                )

            else:

                color = (
                    160,
                    160,
                    160
                )

            cv2.rectangle(
                image,
                (x, y),
                (
                    x + w,
                    y + h
                ),
                color,
                2
            )

            cv2.putText(
                image,
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
        # RIGHT PANEL
        # ====================================================

        panel_width = 500

        panel_height = (
            image.shape[0]
        )

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
            "CONFLICT AGENT",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            panel,
            f"Candidates: {len(candidates)}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (220, 220, 220),
            1,
            cv2.LINE_AA
        )

        # ====================================================
        # AGREEMENT GROUPS
        # ====================================================

        agreement_groups = {

            3: [],

            2: [],

            1: []
        }

        for index, candidate in enumerate(
            candidates,
            start=1
        ):

            evidence_agents = (
                get_evidence_agents(
                    candidate
                )
            )

            evidence_count = len(
                evidence_agents
            )

            if evidence_count in (
                agreement_groups
            ):

                agreement_groups[
                    evidence_count
                ].append(
                    (
                        index,
                        candidate,
                        evidence_agents
                    )
                )

        # ====================================================
        # AGREEMENT
        # ====================================================

        cv2.putText(
            panel,
            "AGREEMENT",
            (20, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        y_position = 150

        # ====================================================
        # CANDIDATE NAME
        # ====================================================

        def candidate_name(
            index,
            candidate
        ):

            class_name = candidate.get(
                "class",
                "unknown"
            )

            if (
                class_name
                and
                str(
                    class_name
                ).lower()
                !=
                "unknown"
            ):

                return (
                    f"C{index}"
                    f" ({class_name})"
                )

            return (
                f"C{index}"
            )

        # ====================================================
        # TEXT WRAPPING
        # ====================================================

        def draw_wrapped_text(
            text,
            x,
            y,
            max_chars=56,
            line_height=25,
            font_scale=0.43
        ):

            words = text.split()

            lines = []

            current = ""

            for word in words:

                test = (
                    f"{current} {word}"
                    if current
                    else word
                )

                if len(test) > max_chars:

                    if current:

                        lines.append(
                            current
                        )

                    current = word

                else:

                    current = test

            if current:

                lines.append(
                    current
                )

            for line in lines:

                cv2.putText(
                    panel,
                    line,
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (220, 220, 220),
                    1,
                    cv2.LINE_AA
                )

                y += line_height

            return y

        # ====================================================
        # 3 AGENT AGREEMENT
        # ====================================================

        three_agent = (
            agreement_groups[3]
        )

        if three_agent:

            entries = []

            for (
                index,
                candidate,
                evidence_agents
            ) in three_agent:

                entries.append(
                    candidate_name(
                        index,
                        candidate
                    )
                )

            text = (
                "3 Agents: "
                +
                ", ".join(
                    entries
                )
            )

        else:

            text = (
                "3 Agents: None"
            )

        y_position = draw_wrapped_text(
            text,
            25,
            y_position
        )

        # ====================================================
        # 2 AGENT AGREEMENT
        # ====================================================

        two_agent = (
            agreement_groups[2]
        )

        if two_agent:

            for (
                index,
                candidate,
                evidence_agents
            ) in two_agent:

                text = (

                    "2 Agents: "

                    +

                    candidate_name(
                        index,
                        candidate
                    )

                    +

                    " ("

                    +

                    ", ".join(
                        evidence_agents
                    )

                    +

                    ")"
                )

                y_position = (
                    draw_wrapped_text(
                        text,
                        25,
                        y_position
                    )
                )

        else:

            y_position = (
                draw_wrapped_text(
                    "2 Agents: None",
                    25,
                    y_position
                )
            )

        # ====================================================
        # 1 AGENT
        # ====================================================

        one_agent = (
            agreement_groups[1]
        )

        if one_agent:

            entries = []

            for (
                index,
                candidate,
                evidence_agents
            ) in one_agent:

                agent_name = (

                    evidence_agents[0]

                    if evidence_agents

                    else "Unknown"
                )

                entries.append(

                    f"{candidate_name(index, candidate)}"
                    f" ({agent_name})"
                )

            text = (
                "1 Agent: "
                +
                ", ".join(
                    entries
                )
            )

            y_position = (
                draw_wrapped_text(
                    text,
                    25,
                    y_position
                )
            )

        else:

            y_position = (
                draw_wrapped_text(
                    "1 Agent: None",
                    25,
                    y_position
                )
            )

        # ====================================================
        # CONFLICT STATISTICS
        # ====================================================

        y_position += 30

        cv2.putText(
            panel,
            "CONFLICT",
            (20, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        y_position += 35

        conflict_states = {

            "LOW": 0,

            "MODERATE": 0,

            "HIGH": 0,

            "UNKNOWN": 0
        }

        for candidate in candidates:

            state = candidate.get(
                "conflict_state",
                "UNKNOWN"
            )

            if state in conflict_states:

                conflict_states[
                    state
                ] += 1

        conflict_lines = [

            f"Low: "
            f"{conflict_states['LOW']}",

            f"Moderate: "
            f"{conflict_states['MODERATE']}",

            f"High: "
            f"{conflict_states['HIGH']}"
        ]

        for line in conflict_lines:

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
        # COMBINE IMAGE + PANEL
        # ====================================================

        final_image = np.hstack([
            image,
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