"""learned_fusion_agent.py - fusion layer driven by the trained meta-classifier (Method 1)."""
import numpy as np
import joblib

from ..fusion.fusion_agent import FusionAgent
from ..fusion.common import (LABELS, STATUS_DETECTED, STATUS_CLEAR, STATUS_REVIEW)
from ..fusion.decision_policy import SUBLABELS
from .features import FEATURE_NAMES, candidate_to_features


class LearnedFusionAgent(FusionAgent):
    """Same interface as FusionAgent (analyze / fuse_candidate / save_visualization)."""

    def __init__(self, model_path, detect_threshold=None, clear_threshold=None):
        super().__init__()
        bundle = joblib.load(model_path)
        if bundle["feature_names"] != FEATURE_NAMES:
            raise ValueError("Model was trained with different features. Rebuild the dataset and retrain.")
        self.model = bundle["model"]
        self.calibrator = bundle["calibrator"]
        self.algo = bundle["algo"]
        self.detect_t = detect_threshold if detect_threshold is not None else bundle["detect_threshold"]
        self.clear_t = clear_threshold if clear_threshold is not None else bundle["clear_threshold"]

    @staticmethod
    def _logit(p):
        p = np.clip(p, 1e-4, 1 - 1e-4)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def predict_proba(self, features):
        raw = self.model.predict_proba(np.array([features]))[:, 1]
        return float(self.calibrator.predict_proba(self._logit(raw))[:, 1][0])

    def _top_drivers(self, features, k=3):
        """Largest contributions for a logistic model; empty for tree models."""
        if self.algo != "lr":
            return []
        scaler, clf = self.model[0], self.model[-1]
        contrib = clf.coef_[0] * scaler.transform(np.array([features]))[0]
        order = np.argsort(-np.abs(contrib))[:k]
        return [f"{FEATURE_NAMES[i]} ({contrib[i]:+.2f})" for i in order]

    def fuse_candidate(self, c):
        features = candidate_to_features(c)
        n_present = features[FEATURE_NAMES.index("n_present")]
        p = self.predict_proba(features) if n_present > 0 else 0.5
        reasons = []

        if n_present == 0:
            status, sub = STATUS_REVIEW, ""
            reasons.append("No agent detected anything for this candidate.")
        elif p >= self.detect_t:
            status, sub = STATUS_DETECTED, SUBLABELS["H0"]
            reasons.append(f"Learned probability {p:.2f} >= {self.detect_t:.2f}.")
        elif p <= self.clear_t:
            status = STATUS_CLEAR
            sub = SUBLABELS.get(c.get("bayes_hypothesis"), "Not camouflaged")
            reasons.append(f"Learned probability {p:.2f} <= {self.clear_t:.2f}.")
        else:
            status, sub = STATUS_REVIEW, ""
            reasons.append(f"Learned probability {p:.2f} is between {self.clear_t:.2f} and {self.detect_t:.2f}.")

        drivers = self._top_drivers(features)
        if drivers:
            reasons.append("Main drivers: " + ", ".join(drivers))

        return {
            "final_status": status,
            "final_label": LABELS[status],
            "final_sublabel": sub,
            "final_suspicion": round(p, 4),
            "final_bbox": c.get("bbox"),
            "final_class": c.get("class"),
            "fusion_method": f"learned-{self.algo}",
            "fusion_signals": {"learned_probability": round(p, 4)},
            "fusion_reasons": reasons,
        }