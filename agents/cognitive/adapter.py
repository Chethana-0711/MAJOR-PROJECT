"""adapter.py - connects the pipeline's candidate dicts to the Bayesian CognitiveAgent."""
from typing import Dict, List
from .common import clip01


def candidate_to_inputs(c: Dict) -> Dict:
    """
    Map one Uncertainty-Agent candidate to CognitiveAgent.reason() arguments.
    Uses the keys the conflict/uncertainty agents really produce:
      spatial_status / spectral_status / feature_status in ("DETECTED", "EVIDENCE_PRESENT")
      spatial_evidence / spectral_evidence / feature_evidence
      conflict, uncertainty
    An agent that did not detect anything becomes None (missing), not 0.0.
    """
    def score(name):
        detected = str(c.get(f"{name}_status", "")).upper() in ("DETECTED", "EVIDENCE_PRESENT")
        return clip01(c.get(f"{name}_evidence", 0.0)) if detected else None

    return dict(
        spatial=score("spatial"),
        spectral=score("spectral"),
        feature=score("feature"),
        conflict=clip01(c.get("conflict", 0.0)),
        uncertainty=clip01(c.get("uncertainty", 0.0)),
    )


def analyze_candidates(agent, candidates: List[Dict]) -> List[Dict]:
    """Run the Bayesian cognitive agent on every candidate; keep all original keys and
    add bayes_* keys (existing 'decision' / 'confidence' keys are never overwritten)."""
    out = []
    for c in candidates:
        r = agent.reason(**candidate_to_inputs(c))
        out.append({
            **c,
            "bayes_decision": r.decision,
            "bayes_hypothesis": r.hypothesis,
            "bayes_camouflage_detected": r.camouflage_detected,
            "bayes_suspicion": r.suspicion_score,
            "bayes_posterior": r.posterior,
            "bayes_flags": r.flags,
            "bayes_explanation": r.explanation,
        })
    return out


def to_legacy_candidate(c: Dict) -> Dict:
    """Map a pipeline candidate to the keys the older rule-based agents/cognitive_agent.py expects
    (it reads spatial_present, thermal_evidence, agent_count ... which the pipeline never sets)."""
    def detected(name):
        return str(c.get(f"{name}_status", "")).upper() in ("DETECTED", "EVIDENCE_PRESENT")

    return {
        **c,
        "spatial_present": detected("spatial"),
        "spectral_present": detected("spectral"),
        "feature_present": detected("feature"),
        "spatial_confidence": c.get("spatial_evidence", 0.0),
        "thermal_evidence": c.get("spectral_evidence", 0.0),
        "feature_evidence": c.get("feature_evidence", 0.0),
        "agent_count": c.get("available_count", 0),
        "uncertainty": c.get("uncertainty", 1.0),
        "conflict": c.get("conflict", 1.0),
    }
