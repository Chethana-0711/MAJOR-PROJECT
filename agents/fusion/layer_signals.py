"""layer_signals.py - turn every earlier layer's output into signals the fusion can combine.

Layers read (all attached to each candidate by the pipeline):
  Spatial / Spectral / Feature : <name>_status, <name>_evidence
  Conflict                     : conflict, agreement, support_count, supporting_agents
  Uncertainty                  : uncertainty, confidence, reliability
  Cognitive (Bayesian)         : bayes_posterior, bayes_hypothesis, bayes_decision, bayes_flags
  Cognitive (rule-based)       : rule_decision, rule_confidence
"""
from typing import Dict, Optional
from .common import AGENTS, RULE_SCORE, FusionConfig, clip01, to_float, is_present


def agent_scores(c: Dict) -> Dict[str, Optional[float]]:
    """Evidence per agent; None when the agent did not detect (missing, not zero)."""
    out = {}
    for name in AGENTS:
        detected = is_present(c.get(f"{name}_status", ""))
        out[name] = clip01(to_float(c.get(f"{name}_evidence"))) if detected else None
    return out


def evidence_signal(c: Dict, cfg: FusionConfig) -> Optional[float]:
    """Spatial + spectral + feature evidence, weighted by agent weight x reliability (1 - uncertainty)."""
    reliability = clip01(1.0 - to_float(c.get("uncertainty"), 1.0))
    num = den = 0.0
    for name, score in agent_scores(c).items():
        if score is None:
            continue
        w = cfg.agent_weights.get(name, 0.0) * max(reliability, 0.05)
        num += w * score
        den += w
    return num / den if den > 0 else None


def bayes_signal(c: Dict) -> Optional[float]:
    post = c.get("bayes_posterior")
    return clip01(post["H0"]) if post and "H0" in post else None


def rule_signal(c: Dict) -> Optional[float]:
    decision = str(c.get("rule_decision", "")).upper()
    if decision not in RULE_SCORE:
        return None
    conf = clip01(to_float(c.get("rule_confidence"), 1.0))
    return 0.5 + (RULE_SCORE[decision] - 0.5) * conf


def agreement_signal(c: Dict, cfg: FusionConfig) -> Optional[float]:
    """Share of the expected agents that support the candidate (conflict layer)."""
    if "support_count" in c:
        support = to_float(c["support_count"])
    elif "supporting_agents" in c:
        support = len(c["supporting_agents"])
    else:
        return None
    return clip01(support / cfg.expected_agents)


def trust_signal(c: Dict, cfg: FusionConfig) -> float:
    """How much the combined score should be believed (uncertainty + conflict layers)."""
    w = cfg.trust_weights
    total = sum(w.values())
    t = (w["uncertainty"] * (1.0 - clip01(to_float(c.get("uncertainty"), 1.0)))
         + w["conflict"] * (1.0 - clip01(to_float(c.get("conflict"), 1.0)))
         + w["confidence"] * clip01(to_float(c.get("confidence"), 0.0)))
    return clip01(t / total)


def compute_signals(c: Dict, cfg: FusionConfig) -> Dict:
    return {
        "evidence": evidence_signal(c, cfg),
        "bayes": bayes_signal(c),
        "rule": rule_signal(c),
        "agreement": agreement_signal(c, cfg),
        "trust": trust_signal(c, cfg),
    }