"""fusion_agent.py - final decision layer. Reads the output of ALL earlier layers."""
import json
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from .common import (FusionConfig, STATUS_DETECTED, STATUS_CLEAR, STATUS_REVIEW,
                     is_present, to_float)
from .layer_signals import compute_signals
from .weighted_fusion import fuse_signals
from .decision_policy import decide

COLOURS = {                      # BGR
    STATUS_DETECTED: (0, 0, 255),
    STATUS_CLEAR: (0, 200, 0),
    "REVIEW": (0, 165, 255),
}


def certification_parameters(r: Dict) -> Dict:
    """The evidence behind one decision, read from the candidate (independent of fusion method)."""
    def evidence(name):
        if is_present(r.get(f"{name}_status", "")):
            return round(to_float(r.get(f"{name}_evidence")), 3)
        return None                                              # agent did not report

    if "support_count" in r:
        support = int(to_float(r["support_count"]))
    else:
        support = len(r.get("supporting_agents") or [])
    post = r.get("bayes_posterior") or {}
    return {
        "spatial_evidence": evidence("spatial"),
        "spectral_evidence": evidence("spectral"),
        "feature_evidence": evidence("feature"),
        "agents_supporting": f"{support}/3",
        "conflict": round(to_float(r.get("conflict")), 3),
        "uncertainty": round(to_float(r.get("uncertainty")), 3),
        "cognitive_p_camouflaged": round(to_float(post.get("H0")), 3) if post else None,
    }


class FusionAgent:
    def __init__(self, config: Optional[FusionConfig] = None):
        self.cfg = config or FusionConfig()

    def fuse_candidate(self, c: Dict) -> Dict:
        signals = compute_signals(c, self.cfg)
        raw, factor, final = fuse_signals(signals, self.cfg)
        status, label, sublabel, reasons = decide(c, signals, final, self.cfg)
        return {
            "final_status": status,
            "final_label": label,
            "final_sublabel": sublabel,
            "final_suspicion": round(final, 4),
            "final_bbox": c.get("bbox"),            # [x, y, w, h] from the conflict agent
            "final_class": c.get("class"),
            "fusion_raw_score": round(raw, 4),
            "fusion_trust_factor": round(factor, 4),
            "fusion_signals": {k: (None if v is None else round(v, 4)) for k, v in signals.items()},
            "fusion_reasons": reasons,
        }

    def analyze(self, candidates: List[Dict]) -> List[Dict]:
        return [{**c, **self.fuse_candidate(c)} for c in candidates]

    # ------------------------------------------------------------------ #
    # Debug view: every candidate (cleared ones as thin outlines)
    # ------------------------------------------------------------------ #
    def save_visualization(self, rgb_path, results: List[Dict], output_path) -> Optional[Path]:
        image = cv2.imread(str(rgb_path))
        if image is None:
            return None
        for i, r in enumerate(results, start=1):
            bbox = r.get("final_bbox")
            if not bbox:
                continue
            x, y, w, h = [int(v) for v in bbox]
            colour = COLOURS.get(r["final_status"], (255, 255, 255))
            if r["final_status"] == STATUS_CLEAR:
                cv2.rectangle(image, (x, y), (x + w, y + h), colour, 1)   # thin box, no label
                continue
            cv2.rectangle(image, (x, y), (x + w, y + h), colour, 3)
            text = f"C{i} {r['final_label']} {r['final_suspicion']:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            top = max(y - th - 8, 0)
            cv2.rectangle(image, (x, top), (x + tw + 6, top + th + 8), colour, -1)
            cv2.putText(image, text, (x + 3, top + th + 2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), image)
        return output_path

    # ------------------------------------------------------------------ #
    # Final certified output: ONLY camouflaged objects, with their evidence
    # ------------------------------------------------------------------ #
    def save_certified_output(self, rgb_path, results: List[Dict], output_path,
                              report_path=None) -> Optional[Path]:
        image = cv2.imread(str(rgb_path))
        if image is None:
            return None

        detected = [r for r in results if r["final_status"] == STATUS_DETECTED and r.get("final_bbox")]
        detected.sort(key=lambda r: -r["final_suspicion"])
        n_review = sum(1 for r in results if r["final_status"] == STATUS_REVIEW)

        # highlight: translucent fill, then thick outline and label
        overlay = image.copy()
        for r in detected:
            x, y, w, h = [int(v) for v in r["final_bbox"]]
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), -1)
        image = cv2.addWeighted(overlay, 0.25, image, 0.75, 0)

        report = []
        for k, r in enumerate(detected, start=1):
            x, y, w, h = [int(v) for v in r["final_bbox"]]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 3)
            text = f"#{k} Camouflaged {r['final_suspicion']:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            top = max(y - th - 10, 0)
            cv2.rectangle(image, (x, top), (x + tw + 8, top + th + 10), (0, 0, 255), -1)
            cv2.putText(image, text, (x + 4, top + th + 3), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 1, cv2.LINE_AA)
            report.append({
                "id": k,
                "bbox_xywh": [x, y, w, h],
                "detector_class": r.get("final_class"),
                "decision": r["final_status"],
                "confidence": round(float(r["final_suspicion"]), 4),
                "parameters": certification_parameters(r),
                "rule": (r.get("fusion_reasons") or [""])[0],
                "fusion_method": r.get("fusion_method", "weighted"),
            })

        # side panel with the evidence for every highlighted object
        panel_w, line_h, block_h = 440, 22, 250
        height = max(image.shape[0], 90 + max(len(report), 1) * block_h + 40)
        canvas = np.full((height, image.shape[1] + panel_w, 3), 28, np.uint8)
        canvas[:image.shape[0], :image.shape[1]] = image
        px = image.shape[1] + 18
        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(canvas, "CAMOUFLAGED OBJECTS", (px, 34), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"Found: {len(report)}", (px, 60), font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        y = 95
        if not report:
            cv2.putText(canvas, "No camouflaged object detected.", (px, y), font, 0.55,
                        (220, 220, 220), 1, cv2.LINE_AA)
        for item in report:
            p = item["parameters"]
            cv2.putText(canvas, f"#{item['id']}  CAMOUFLAGED", (px, y), font, 0.6, (80, 80, 255), 2, cv2.LINE_AA)
            y += line_h
            bar_w = panel_w - 40                              # confidence bar
            cv2.rectangle(canvas, (px, y - 12), (px + bar_w, y + 4), (90, 90, 90), 1)
            cv2.rectangle(canvas, (px, y - 12), (px + int(bar_w * item["confidence"]), y + 4), (0, 0, 255), -1)
            y += line_h
            rows = [
                ("Confidence", f"{item['confidence']:.2f}"),
                ("Spatial (YOLO)", "not reported" if p["spatial_evidence"] is None else f"{p['spatial_evidence']:.2f}"),
                ("Spectral (thermal)", "not reported" if p["spectral_evidence"] is None else f"{p['spectral_evidence']:.2f}"),
                ("Feature (texture)", "not reported" if p["feature_evidence"] is None else f"{p['feature_evidence']:.2f}"),
                ("Agents supporting", p["agents_supporting"]),
                ("Conflict / Uncertainty", f"{p['conflict']:.2f} / {p['uncertainty']:.2f}"),
                ("Cognitive P(camo)", "n/a" if p["cognitive_p_camouflaged"] is None else f"{p['cognitive_p_camouflaged']:.2f}"),
                ("Rule", item["rule"][:34]),
            ]
            for name, value in rows:
                cv2.putText(canvas, name, (px, y), font, 0.45, (170, 170, 170), 1, cv2.LINE_AA)
                cv2.putText(canvas, str(value), (px + 205, y), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                y += line_h
            y += 24

        if n_review:
            cv2.putText(canvas, f"{n_review} uncertain candidate(s) not shown", (px, height - 18),
                        font, 0.42, (150, 150, 150), 1, cv2.LINE_AA)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), canvas)

        if report_path is not None:
            Path(report_path).write_text(json.dumps({
                "image": str(rgb_path),
                "camouflaged_objects": report,
                "uncertain_candidates_not_shown": n_review,
            }, indent=2, default=str), encoding="utf-8")
        return output_path