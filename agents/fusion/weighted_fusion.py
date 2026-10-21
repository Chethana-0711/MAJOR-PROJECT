"""weighted_fusion.py - combine the layer signals into one suspicion score."""
from typing import Dict, Tuple
from .common import FusionConfig, clip01


def fuse_signals(signals: Dict, cfg: FusionConfig) -> Tuple[float, float, float]:
    """
    raw   = weighted mean of the available belief signals (missing signals are dropped and
            the weights renormalised - a missing layer is not treated as a zero)
    trust factor = floor + (1 - floor) * trust       (uncertainty / conflict / confidence layers)
    final = 0.5 + (raw - 0.5) * trust factor         (low trust pulls the score toward 0.5)
    Returns (raw, trust_factor, final).
    """
    num = den = 0.0
    for name, w in cfg.signal_weights.items():
        value = signals.get(name)
        if value is None:
            continue
        num += w * value
        den += w
    raw = num / den if den > 0 else 0.5

    factor = cfg.trust_floor + (1.0 - cfg.trust_floor) * signals["trust"]
    final = clip01(0.5 + (raw - 0.5) * factor)
    return raw, factor, final