"""calibration.py - fit p(score | Hk) from labelled data (use in evaluate.py)."""
import math
from typing import Dict, List
from .common import HYPS, Gauss, DEFAULT_MODELS


def fit_models(samples: Dict[str, List[float]], labels: Dict[str, List[str]]):
    """
    samples[agent] = list of scores
    labels[agent]  = list of "H0" / "H1" / "H2" ground-truth labels (same length)
    Falls back to DEFAULT_MODELS when a hypothesis has fewer than 2 samples.
    """
    fitted = {}
    for agent, xs in samples.items():
        fitted[agent] = {}
        for h in HYPS:
            v = [x for x, y in zip(xs, labels[agent]) if y == h]
            if len(v) < 2:
                fitted[agent][h] = DEFAULT_MODELS.get(agent, DEFAULT_MODELS["spatial"])[h]
                continue
            mu = sum(v) / len(v)
            sd = math.sqrt(sum((x - mu) ** 2 for x in v) / (len(v) - 1))
            fitted[agent][h] = Gauss(mu, max(sd, 0.05))
    return fitted