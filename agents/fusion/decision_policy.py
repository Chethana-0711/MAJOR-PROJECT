"""decision_policy.py - turn the fused score into Detected / Not Detected / Review."""
from typing import Dict, List, Tuple
from .common import (FusionConfig, RULE_SCORE, LABELS,
                     STATUS_DETECTED, STATUS_CLEAR, STATUS_REVIEW)

SUBLABELS = {"H0": "Camouflaged object", "H1": "Ordinary object", "H2": "No object"}


def _bayes_vote(c: Dict) -> int:
    hyp = c.get("bayes_hypothesis")
    return 1 if hyp == "H0" else (-1 if hyp in ("H1", "H2") else 0)


def _rule_vote(c: Dict) -> int:
    score = RULE_SCORE.get(str(c.get("rule_decision", "")).upper())
    if score is None:
        return 0
    return 1 if score >= 0.65 else (-1 if score <= 0.35 else 0)


def decide(c: Dict, signals: Dict, final: float, cfg: FusionConfig) -> Tuple[str, str, str, List[str]]:
    """Returns (status, label, sublabel, reasons). Review gates run before the thresholds."""
    reasons: List[str] = []

    # Gate 1 - nothing to fuse
    belief = [signals.get(k) for k in ("evidence", "bayes", "rule", "agreement")]
    if all(v is None for v in belief):
        reasons.append("No layer produced usable evidence.")
        return STATUS_REVIEW, LABELS[STATUS_REVIEW], "", reasons

    # Gate 2 - the Bayesian cognitive layer already sent it to review
    if str(c.get("bayes_decision", "")).startswith("UNCERTAIN"):
        flags = c.get("bayes_flags") or []
        reasons.append(f"Bayesian cognitive layer requested review (flags: {flags or 'none'}).")
        return STATUS_REVIEW, LABELS[STATUS_REVIEW], "", reasons

    # Gate 3 - the two cognitive methods disagree
    if _bayes_vote(c) * _rule_vote(c) == -1:
        reasons.append("Bayesian and rule-based cognitive layers disagree.")
        return STATUS_REVIEW, LABELS[STATUS_REVIEW], "", reasons

    # Thresholds on the fused score
    if final >= cfg.detect_threshold:
        reasons.append(f"Fused suspicion {final:.2f} >= {cfg.detect_threshold:.2f}.")
        return STATUS_DETECTED, LABELS[STATUS_DETECTED], SUBLABELS["H0"], reasons

    if final <= cfg.clear_threshold:
        sub = SUBLABELS.get(c.get("bayes_hypothesis"), "Not camouflaged")
        reasons.append(f"Fused suspicion {final:.2f} <= {cfg.clear_threshold:.2f}.")
        return STATUS_CLEAR, LABELS[STATUS_CLEAR], sub, reasons

    reasons.append(
        f"Fused suspicion {final:.2f} is between {cfg.clear_threshold:.2f} and {cfg.detect_threshold:.2f}.")
    return STATUS_REVIEW, LABELS[STATUS_REVIEW], "", reasons