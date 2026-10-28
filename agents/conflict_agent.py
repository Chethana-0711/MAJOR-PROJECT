from pathlib import Path

import cv2
import numpy as np


class ConflictAgent:
    """Match spatial, thermal, and multimodal-CNN evidence by bounding box."""

    def __init__(self, iou_threshold=0.10, containment_threshold=0.30,
                 center_similarity_threshold=0.70):
        self.iou_threshold = iou_threshold
        self.containment_threshold = containment_threshold
        self.center_similarity_threshold = center_similarity_threshold

    @staticmethod
    def _box(box):
        """Convert a bbox to x, y, width, height."""
        if not box or len(box) != 4:
            return [0, 0, 0, 0]
        return [float(v) for v in box]

    @staticmethod
    def _intersection(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        w = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
        h = max(0.0, min(ay + ah, by + bh) - max(ay, by))
        return w * h

    def regions_match(self, box_a, box_b):
        a, b = self._box(box_a), self._box(box_b)
        intersection = self._intersection(a, b)
        area_a, area_b = a[2] * a[3], b[2] * b[3]
        union = area_a + area_b - intersection
        iou = intersection / union if union > 0 else 0.0
        smaller = min(area_a, area_b)
        containment = intersection / smaller if smaller > 0 else 0.0

        center_a = (a[0] + a[2] / 2, a[1] + a[3] / 2)
        center_b = (b[0] + b[2] / 2, b[1] + b[3] / 2)
        distance = np.hypot(center_a[0] - center_b[0],
                            center_a[1] - center_b[1])
        reference = max(np.hypot(a[2], a[3]), np.hypot(b[2], b[3]), 1.0)
        center_similarity = 1.0 - min(distance / reference, 1.0)

        matched = (iou >= self.iou_threshold or
                   containment >= self.containment_threshold or
                   center_similarity >= self.center_similarity_threshold)
        return {
            "matched": matched,
            "iou": float(iou),
            "containment": float(containment),
            "center_similarity": float(center_similarity),
            "match_score": float(max(iou, containment, center_similarity)),
        }

    @staticmethod
    def union_boxes(box_a, box_b):
        ax, ay, aw, ah = [float(v) for v in box_a]
        bx, by, bw, bh = [float(v) for v in box_b]
        x1, y1 = min(ax, bx), min(ay, by)
        x2, y2 = max(ax + aw, bx + bw), max(ay + ah, by + bh)
        return [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]

    @staticmethod
    def _regions(result, key):
        if result is None:
            return []
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            return list(result.get(key, []) or [])
        return []

    def _feature_regions(self, result):
        regions = self._regions(result, "regions")
        localized = self._regions(result, "localized_evidence_regions")
        # Localized evidence is retained for context, but isn't a full detection.
        return regions + [
            {**region, "evidence_type": "localized"} for region in localized
        ]

    @staticmethod
    def _evidence(region, keys):
        if region is None:
            return 0.0
        for key in keys:
            if key in region:
                try:
                    return float(region[key])
                except (TypeError, ValueError):
                    return 0.0
        return 0.0

    def _build_candidates(self, spatial, spectral, feature):
        candidates = [
            {"bbox": self._box(r.get("bbox", [0, 0, 0, 0])),
             "class": r.get("class", "unknown"),
             "spatial": r, "spectral": None, "feature": None}
            for r in spatial
        ]

        # Match each spatial region to at most one thermal region, strongest match first.
        matches = []
        for si, candidate in enumerate(candidates):
            for ti, thermal in enumerate(spectral):
                match = self.regions_match(candidate["bbox"], thermal.get("bbox"))
                if match["matched"]:
                    matches.append((match["match_score"], si, ti))
        used_spatial, used_thermal = set(), set()
        for _, si, ti in sorted(matches, reverse=True):
            if si in used_spatial or ti in used_thermal:
                continue
            candidates[si]["spectral"] = spectral[ti]
            candidates[si]["bbox"] = self.union_boxes(
                candidates[si]["bbox"], spectral[ti]["bbox"])
            used_spatial.add(si)
            used_thermal.add(ti)

        # Preserve thermal regions not matched to a spatial detection.
        for ti, thermal in enumerate(spectral):
            if ti not in used_thermal:
                candidates.append({
                    "bbox": self._box(thermal.get("bbox", [0, 0, 0, 0])),
                    "class": "unknown", "spatial": None,
                    "spectral": thermal, "feature": None,
                })

        # Match feature regions to existing candidates, strongest match first.
        matches = []
        for fi, item in enumerate(feature):
            for ci, candidate in enumerate(candidates):
                match = self.regions_match(candidate["bbox"], item.get("bbox"))
                if match["matched"]:
                    matches.append((match["match_score"], fi, ci))
        used_features, used_candidates = set(), set()
        for _, fi, ci in sorted(matches, reverse=True):
            if fi in used_features or ci in used_candidates:
                continue
            candidates[ci]["feature"] = feature[fi]
            candidates[ci]["bbox"] = self.union_boxes(
                candidates[ci]["bbox"], feature[fi]["bbox"])
            used_features.add(fi)
            used_candidates.add(ci)

        # Preserve unmatched feature regions as independent candidates.
        for fi, item in enumerate(feature):
            if fi not in used_features:
                candidates.append({
                    "bbox": self._box(item.get("bbox", [0, 0, 0, 0])),
                    "class": "unknown", "spatial": None,
                    "spectral": None, "feature": item,
                })
        return candidates

    def _analyze_candidate(self, candidate, available):
        spatial, spectral, feature = (
            candidate["spatial"], candidate["spectral"], candidate["feature"])
        regions = {"spatial": spatial, "spectral": spectral, "feature": feature}
        statuses, evidence = {}, {}

        for name, region in regions.items():
            if region is not None:
                statuses[name] = (
                    "EVIDENCE_PRESENT"
                    if name == "feature" and region.get("evidence_type") == "localized"
                    else "DETECTED"
                )
            else:
                statuses[name] = "NO_DETECTION" if available[name] else "NOT_AVAILABLE"

        evidence["spatial"] = self._evidence(spatial, ("confidence",))
        evidence["spectral"] = self._evidence(
            spectral, ("combined_evidence", "evidence", "ml_probability"))
        evidence["feature"] = self._evidence(feature, ("evidence",))

        supporting = [
            name for name, region in regions.items()
            if region is not None and not (
                name == "feature" and region.get("evidence_type") == "localized")
        ]
        available_agents = [name for name in regions if available[name]]
        missing = [name for name in regions if not available[name]]
        non_detecting = [name for name in available_agents if name not in supporting]
        available_evidence = [evidence[name] for name in available_agents]
        agreement = len(supporting) / len(available_agents) if available_agents else 0.0
        conflict = (max(available_evidence) - min(available_evidence)
                    if len(available_evidence) >= 2 else 0.0)

        if not supporting:
            pattern = "NO DETECTION"
        elif len(supporting) == len(available_agents) and len(available_agents) == 3:
            pattern = "TRIPLE AGREEMENT"
        elif len(supporting) == len(available_agents):
            pattern = "FULL AVAILABLE AGREEMENT"
        elif len(supporting) == 1:
            pattern = f"{supporting[0].upper()}-ONLY EVIDENCE"
        else:
            pattern = "PARTIAL AGREEMENT"

        if len(available_agents) < 2:
            conflict_state = "UNKNOWN"
        elif conflict < 0.20:
            conflict_state = "LOW"
        elif conflict < 0.50:
            conflict_state = "MODERATE"
        else:
            conflict_state = "HIGH"

        return {
            "bbox": candidate["bbox"], "class": candidate["class"],
            "spatial_status": statuses["spatial"],
            "spectral_status": statuses["spectral"],
            "feature_status": statuses["feature"],
            "spatial_evidence": evidence["spatial"],
            "spectral_evidence": evidence["spectral"],
            "feature_evidence": evidence["feature"],
            "available_agents": available_agents,
            "available_count": len(available_agents),
            "missing_agents": missing,
            "supporting_agents": supporting,
            "support_count": len(supporting),
            "non_detecting_agents": non_detecting,
            "agreement": float(agreement),
            "conflict": float(np.clip(conflict, 0.0, 1.0)),
            "conflict_state": conflict_state,
            "average_evidence": float(np.mean(available_evidence))
                if available_evidence else 0.0,
            "evidence_pattern": pattern,
        }

    def analyze(self, spatial_result=None, spectral_result=None, feature_result=None):
        """Compare outputs from the three agents; any agent may return None."""
        available = {
            "spatial": spatial_result is not None,
            "spectral": spectral_result is not None,
            "feature": feature_result is not None,
        }
        spatial = self._regions(spatial_result, "detections")
        spectral = self._regions(spectral_result, "regions")
        feature = self._feature_regions(feature_result)
        candidates = self._build_candidates(spatial, spectral, feature)
        results = [self._analyze_candidate(candidate, available)
                   for candidate in candidates]
        available_agents = [name for name, yes in available.items() if yes]

        return {
            "agent": "conflict",
            "available_agents": available_agents,
            "available_agent_count": len(available_agents),
            "candidates": results,
            "candidate_count": len(results),
        }

    def save_visualization(self, rgb_path, results,
                           output_path="outputs/conflict_analysis.png"):
        """Draw candidate boxes and a compact summary beside the RGB image."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image = cv2.imread(str(rgb_path))
        if image is None:
            raise ValueError(f"Could not load image: {rgb_path}")

        candidates = results.get("candidates", [])
        for index, item in enumerate(candidates, 1):
            box = item.get("bbox", [0, 0, 0, 0])
            x, y, w, h = [int(round(float(v))) for v in box]
            x = max(0, min(x, image.shape[1] - 1))
            y = max(0, min(y, image.shape[0] - 1))
            w = max(1, min(w, image.shape[1] - x))
            h = max(1, min(h, image.shape[0] - y))
            count = item.get("support_count", 0)
            color = (0, 200, 0) if count >= 3 else (
                (0, 220, 220) if count == 2 else
                ((0, 0, 255) if count == 1 else (160, 160, 160)))
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, f"C{index}", (x, max(20, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        panel = np.zeros((image.shape[0], 440, 3), dtype=np.uint8)
        lines = [
            "CONFLICT AGENT",
            f"Candidates: {len(candidates)}",
            "",
            "AGREEMENT / SUPPORT",
            f"3 agents: {sum(c['support_count'] == 3 for c in candidates)}",
            f"2 agents: {sum(c['support_count'] == 2 for c in candidates)}",
            f"1 agent:  {sum(c['support_count'] == 1 for c in candidates)}",
            "",
            "CONFLICT LEVELS",
        ]
        for state in ("LOW", "MODERATE", "HIGH", "UNKNOWN"):
            lines.append(
                f"{state.title()}: " +
                str(sum(c.get("conflict_state") == state for c in candidates)))
        y = 35
        for line in lines:
            cv2.putText(panel, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (230, 230, 230), 1, cv2.LINE_AA)
            y += 30

        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), np.hstack((image, panel)))
        return output_path
