"""
simple_fusion.py - a simple, transparent fusion layer.

The DETECTORS decide. Weights (they add up to 1.0):
    spatial   0.35   YOLO evidence
    spectral  0.25   thermal evidence
    feature   0.20   edge / texture evidence
    agreement 0.10   how many agents support the candidate (conflict layer)
    cognitive 0.05   Bayesian P(camouflaged)  <- smallest weight
    quality   0.05   1 - uncertainty and 1 - conflict (uncertainty + conflict layers)

An agent that did not report evidence contributes 0 (it is NOT averaged away), so a candidate
backed by only one weak agent cannot reach a high score.
    score >= detect_threshold  -> CAMOUFLAGE DETECTED
    score <= clear_threshold   -> NOT DETECTED
    otherwise                  -> REVIEW
"""
from dataclasses import dataclass, field
from typing import Dict

from .common import (LABELS, STATUS_DETECTED, STATUS_CLEAR, STATUS_REVIEW,
                     clip01, to_float, is_present)
from .fusion_agent import FusionAgent


@dataclass
class SimpleFusionConfig:
    weights: Dict[str, float] = field(default_factory=lambda: {
        "spatial": 0.35, "spectral": 0.25, "feature": 0.20,
        "agreement": 0.10, "cognitive": 0.05, "quality": 0.05})
    detect_threshold: float = 0.45
    clear_threshold: float = 0.25


class SimpleFusionAgent(FusionAgent):
    def __init__(self, config: SimpleFusionConfig = None):
        super().__init__()
        self.simple = config or SimpleFusionConfig()

    def signals(self, c: Dict) -> Dict[str, float]:
        def evidence(name):
            if is_present(c.get(f"{name}_status", "")):
                return clip01(to_float(c.get(f"{name}_evidence")))
            return 0.0

        if "support_count" in c:
            support = to_float(c["support_count"])
        else:
            support = float(len(c.get("supporting_agents") or []))

        post = c.get("bayes_posterior") or {}
        quality = 0.5 * ((1 - clip01(to_float(c.get("uncertainty"), 1.0)))
                         + (1 - clip01(to_float(c.get("conflict"), 1.0))))
        return {
            "spatial": evidence("spatial"),
            "spectral": evidence("spectral"),
            "feature": evidence("feature"),
            "agreement": clip01(support / 3.0),
            "cognitive": clip01(to_float(post.get("H0"), 0.0)),
            "quality": quality,
        }

    def score(self, c: Dict):
        sig = self.signals(c)
        return sum(self.simple.weights[k] * sig[k] for k in self.simple.weights), sig

    def fuse_candidate(self, c: Dict) -> Dict:
        cfg = self.simple
        score, sig = self.score(c)
        if score >= cfg.detect_threshold:
            status, sub = STATUS_DETECTED, "Camouflaged object"
            reason = f"Score {score:.2f} >= {cfg.detect_threshold:.2f}."
        elif score <= cfg.clear_threshold:
            status, sub = STATUS_CLEAR, "Low evidence"
            reason = f"Score {score:.2f} <= {cfg.clear_threshold:.2f}."
        else:
            status, sub = STATUS_REVIEW, ""
            reason = f"Score {score:.2f} is between {cfg.clear_threshold:.2f} and {cfg.detect_threshold:.2f}."
        return {
            "final_status": status,
            "final_label": LABELS[status],
            "final_sublabel": sub,
            "final_suspicion": round(score, 4),
            "final_bbox": c.get("bbox"),
            "final_class": c.get("class"),
            "fusion_method": "simple",
            "fusion_signals": {k: round(v, 3) for k, v in sig.items()},
            "fusion_reasons": [reason],
        }