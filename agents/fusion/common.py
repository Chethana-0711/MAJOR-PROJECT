"""common.py - shared helpers, constants and configuration for the fusion layer."""
from dataclasses import dataclass, field
from typing import Dict

AGENTS = ["spatial", "spectral", "feature"]

# How strongly each rule-based decision (agents/cognitive_agent.py) supports "camouflaged".
RULE_SCORE = {
    "HIGH CONFIDENCE TARGET": 1.00,
    "LIKELY TARGET": 0.80,
    "POTENTIAL CAMOUFLAGED TARGET": 0.65,
    "THERMAL ANOMALY": 0.50,
    "FEATURE ANOMALY": 0.50,
    "UNCERTAIN CANDIDATE": 0.50,
    "UNSUPPORTED SPATIAL DETECTION": 0.35,
    "LIKELY FALSE POSITIVE": 0.10,
}

STATUS_DETECTED = "CAMOUFLAGE DETECTED"
STATUS_CLEAR = "NOT DETECTED"
STATUS_REVIEW = "REVIEW"

LABELS = {
    STATUS_DETECTED: "Camouflaged",
    STATUS_CLEAR: "Not Camouflaged",
    STATUS_REVIEW: "Needs Review",
}


PRESENT_STATUSES = ("DETECTED", "EVIDENCE_PRESENT")


def is_present(status) -> bool:
    """The conflict agent marks an agent as present with DETECTED, or EVIDENCE_PRESENT
    (localized feature evidence). NO_DETECTION / NOT_AVAILABLE mean missing."""
    return str(status).upper() in PRESENT_STATUSES


def clip01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def to_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


@dataclass
class FusionConfig:
    # relative importance of each detector when their evidence is averaged
    agent_weights: Dict[str, float] = field(default_factory=lambda: {
        "spatial": 0.40, "spectral": 0.35, "feature": 0.25})

    # relative importance of each layer-level signal in the weighted fusion
    #   evidence  : spatial + spectral + feature evidence, reliability weighted
    #   bayes     : P(H0) from the Bayesian cognitive layer
    #   rule      : rule-based cognitive decision
    #   agreement : how many agents support the candidate (conflict layer)
    signal_weights: Dict[str, float] = field(default_factory=lambda: {
        "evidence": 0.25, "bayes": 0.40, "rule": 0.15, "agreement": 0.20})

    # trust = how much the fused score is believed (uncertainty + conflict layers)
    trust_weights: Dict[str, float] = field(default_factory=lambda: {
        "uncertainty": 0.40, "conflict": 0.30, "confidence": 0.30})
    trust_floor: float = 0.30        # even at zero trust, keep 30% of the score's distance from 0.5

    detect_threshold: float = 0.60   # suspicion >= this  -> CAMOUFLAGE DETECTED
    clear_threshold: float = 0.40    # suspicion <= this  -> NOT DETECTED
    expected_agents: int = 3