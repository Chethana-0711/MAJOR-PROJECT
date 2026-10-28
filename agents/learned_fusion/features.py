"""features.py - turn one pipeline candidate (output of ALL earlier layers) into a numeric feature vector."""
import math
from typing import Dict, List

from ..fusion.common import clip01, to_float

AGENTS = ["spatial", "spectral", "feature"]
CONFLICT_LEVEL = {"LOW": 0.0, "MODERATE": 1.0, "HIGH": 2.0, "UNKNOWN": 1.0}
RELIABILITY_LEVEL = {"LOW": 0.0, "MEDIUM": 1.0, "HIGH": 2.0}

FEATURE_NAMES: List[str] = [
    # Spatial / Spectral / Feature layers
    "spatial_present", "spatial_evidence",
    "spectral_present", "spectral_evidence",
    "feature_present", "feature_evidence",
    "n_present", "mean_evidence", "max_evidence", "min_evidence", "evidence_spread",
    # Conflict layer
    "conflict", "support_count", "agreement_ratio", "conflict_level",
    # Uncertainty layer
    "uncertainty", "confidence", "reliability_level",
    # Cognitive layer (Bayesian)
    "bayes_h0", "bayes_h1", "bayes_h2", "bayes_review", "bayes_n_flags",
    # Candidate geometry
    "log_area", "aspect_ratio",
]


def candidate_to_features(c: Dict) -> List[float]:
    present, evidence = {}, {}
    for name in AGENTS:
        detected = str(c.get(f"{name}_status", "")).upper() in ("DETECTED", "EVIDENCE_PRESENT")
        present[name] = 1.0 if detected else 0.0
        evidence[name] = clip01(to_float(c.get(f"{name}_evidence"))) if detected else 0.0

    used = [evidence[n] for n in AGENTS if present[n] > 0]
    n_present = float(len(used))
    mean_e = sum(used) / len(used) if used else 0.0
    max_e = max(used) if used else 0.0
    min_e = min(used) if used else 0.0

    if "support_count" in c:
        support = to_float(c["support_count"])
    else:
        support = float(len(c.get("supporting_agents") or []))

    post = c.get("bayes_posterior") or {}
    flags = c.get("bayes_flags") or []

    bbox = c.get("bbox") or [0, 0, 1, 1]
    w, h = max(to_float(bbox[2], 1.0), 1.0), max(to_float(bbox[3], 1.0), 1.0)

    row = {
        "spatial_present": present["spatial"], "spatial_evidence": evidence["spatial"],
        "spectral_present": present["spectral"], "spectral_evidence": evidence["spectral"],
        "feature_present": present["feature"], "feature_evidence": evidence["feature"],
        "n_present": n_present, "mean_evidence": mean_e, "max_evidence": max_e,
        "min_evidence": min_e, "evidence_spread": max_e - min_e,
        "conflict": clip01(to_float(c.get("conflict"), 1.0)),
        "support_count": support,
        "agreement_ratio": clip01(support / len(AGENTS)),
        "conflict_level": CONFLICT_LEVEL.get(str(c.get("conflict_state", "UNKNOWN")).upper(), 1.0),
        "uncertainty": clip01(to_float(c.get("uncertainty"), 1.0)),
        "confidence": clip01(to_float(c.get("confidence"), 0.0)),
        "reliability_level": RELIABILITY_LEVEL.get(str(c.get("reliability", "LOW")).upper(), 0.0),
        "bayes_h0": clip01(to_float(post.get("H0"), 1 / 3)),
        "bayes_h1": clip01(to_float(post.get("H1"), 1 / 3)),
        "bayes_h2": clip01(to_float(post.get("H2"), 1 / 3)),
        "bayes_review": 1.0 if str(c.get("bayes_decision", "")).startswith("UNCERTAIN") else 0.0,
        "bayes_n_flags": float(len(flags)),
        "log_area": math.log(w * h),
        "aspect_ratio": w / h,
    }
    return [float(row[name]) for name in FEATURE_NAMES]
