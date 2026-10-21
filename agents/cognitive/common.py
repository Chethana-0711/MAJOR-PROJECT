"""common.py - shared constants, helpers, data structures and default models."""
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

EPS = 1e-9
HYPS = ["H0", "H1", "H2"]
NAMES = {
    "H0": "CAMOUFLAGED OBJECT",
    "H1": "ORDINARY OBJECT (not camouflaged)",
    "H2": "NO OBJECT",
}


def clip01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def log_norm_pdf(x: float, mu: float, sd: float) -> float:
    sd = max(sd, 1e-3)
    return -0.5 * ((x - mu) / sd) ** 2 - math.log(sd * math.sqrt(2 * math.pi))


def logsumexp(vals: List[float]) -> float:
    m = max(vals)
    return m + math.log(sum(math.exp(v - m) for v in vals))


def normalize_log(logp: Dict[str, float]) -> Dict[str, float]:
    z = logsumexp(list(logp.values()))
    return {h: math.exp(v - z) for h, v in logp.items()}


def chi2_sf(x: float, df: int) -> float:
    """P(Chi2_df >= x) via the regularized lower incomplete gamma series."""
    if x <= 0:
        return 1.0
    a, y = df / 2.0, x / 2.0
    term = total = 1.0 / a
    for n in range(1, 500):
        term *= y / (a + n)
        total += term
        if term < 1e-12 * total:
            break
    lower = math.exp(a * math.log(y) - y - math.lgamma(a)) * total
    return max(0.0, min(1.0, 1.0 - lower))


def entropy_norm(p: Dict[str, float]) -> float:
    h = -sum(v * math.log(v + EPS) for v in p.values())
    return h / math.log(len(p))


@dataclass
class Evidence:
    name: str            # "spatial", "spectral", "feature"
    score: float         # [0, 1]
    reliability: float   # 1 - uncertainty, [0, 1]


@dataclass
class Gauss:
    mu: float
    sd: float


# p(score | Hk) per agent. ASSUMPTIONS - replace using calibration.fit_models().
#   spatial  = YOLO objectness confidence
#   spectral = thermal / spectral anomaly evidence
#   feature  = FeatureAgent evidence (higher = stronger edge/texture anomaly, as in the repo)
DEFAULT_MODELS: Dict[str, Dict[str, Gauss]] = {
    "spatial":  {"H0": Gauss(0.55, 0.20), "H1": Gauss(0.85, 0.10), "H2": Gauss(0.12, 0.10)},
    "spectral": {"H0": Gauss(0.75, 0.15), "H1": Gauss(0.45, 0.20), "H2": Gauss(0.12, 0.10)},
    "feature":  {"H0": Gauss(0.75, 0.15), "H1": Gauss(0.50, 0.20), "H2": Gauss(0.12, 0.10)},
}


@dataclass
class CognitiveResult:
    decision: str
    hypothesis: Optional[str]
    camouflage_detected: bool
    suspicion_score: float
    posterior: Dict[str, float]
    raw_posterior: Dict[str, float]
    test: dict = field(default_factory=dict)
    belief_trace: List[dict] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)
    explanation: List[str] = field(default_factory=list)