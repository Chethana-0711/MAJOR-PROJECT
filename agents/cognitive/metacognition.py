"""metacognition.py - Stage 3: rule-based monitoring of the reasoning process."""
from typing import Dict, List
from .common import HYPS, Evidence, clip01, entropy_norm


def metacognitive_monitor(post: Dict[str, float], trace: List[dict], evidence: List[Evidence],
                          conflict: float, uncertainty: float, test: dict,
                          expected_agents: int = 3):
    """
    Audits the reasoning, not the image.
    Returns (adjusted_posterior, force_review, flags, notes).
    Fired rules shrink the posterior toward uniform and may force review.
    """
    flags, notes = [], []
    shrink, force_review = 0.0, False
    ranked = sorted(HYPS, key=lambda h: post[h], reverse=True)
    top, second = ranked[0], ranked[1]
    margin = post[top] - post[second]

    # R1 conflict between agents
    if conflict >= 0.6:
        flags.append("HIGH_CONFLICT"); shrink += 0.35
        notes.append(f"Agents disagree strongly (conflict={conflict:.2f}).")
        if conflict >= 0.8:
            force_review = True
    elif conflict >= 0.35:
        flags.append("MODERATE_CONFLICT"); shrink += 0.15

    # R2 unreliable evidence
    if uncertainty >= 0.6:
        flags.append("HIGH_UNCERTAINTY"); shrink += 0.30
        notes.append(f"Evidence is unreliable (uncertainty={uncertainty:.2f}).")
    elif uncertainty >= 0.4:
        flags.append("MODERATE_UNCERTAINTY"); shrink += 0.10

    # R3 missing agents
    if len(evidence) < expected_agents:
        flags.append("MISSING_EVIDENCE"); shrink += 0.10 * (expected_agents - len(evidence))
        notes.append(f"Only {len(evidence)}/{expected_agents} agents reported.")

    # R4 one agent dominates
    spreads = [t["spread"] for t in trace if t["step"] == "update"]
    tot = sum(spreads)
    if len(spreads) > 1 and tot > 0 and max(spreads) / tot > 0.75:
        flags.append("SINGLE_AGENT_DOMINATES"); shrink += 0.10
        notes.append(f"One agent supplies {max(spreads) / tot:.0%} of the discriminating evidence.")

    # R5 hypothesis test vs Bayesian posterior disagree
    if test["best"] != top:
        flags.append("METHOD_DISAGREEMENT"); shrink += 0.15
        notes.append(f"Likelihood test favours {test['best']} but posterior favours {top}.")

    # R6 hypothesis test not decisive
    if not test["supported"]:
        flags.append("WEAK_BAYES_FACTOR"); shrink += 0.10
        notes.append("Best hypothesis does not clearly beat the runner-up.")

    # R7 evidence fits none of the hypotheses
    if not test["consistent"]:
        flags.append("NO_HYPOTHESIS_FITS"); shrink += 0.20; force_review = True
        notes.append("Evidence is inconsistent with all three hypotheses (possible novel case).")

    # R8 camouflaged vs ordinary confusion
    if {top, second} == {"H0", "H1"} and margin < 0.3:
        flags.append("CAMO_VS_ORDINARY_CONFUSION"); force_review = True
        notes.append("Cannot separate camouflaged from ordinary object.")

    # R9 flat posterior / narrow margin
    if entropy_norm(post) > 0.85:
        flags.append("HIGH_ENTROPY"); force_review = True
    if margin < 0.15:
        flags.append("NARROW_MARGIN"); force_review = True

    # R10 overconfidence
    if post[top] > 0.95 and (conflict > 0.4 or uncertainty > 0.4):
        flags.append("OVERCONFIDENT"); shrink += 0.15
        notes.append("Belief is extreme despite shaky evidence; tempered.")

    shrink = clip01(shrink)
    uniform = 1.0 / len(HYPS)
    adjusted = {h: (1 - shrink) * post[h] + shrink * uniform for h in HYPS}
    return adjusted, force_review, flags, notes