"""
evaluate.py - compare fusion methods on a held-out candidate CSV (build it from the TEST split).

    python -m agents.learned_fusion.evaluate --data data/fusion/test.csv --model models/learned_fusion.joblib

Methods compared (candidate level, positive = candidate matches a ground-truth target):
  bayes_only   : P(H0) from the Bayesian cognitive layer
  weighted     : reliability-weighted fusion (agents/fusion)
  learned      : trained meta-classifier
"""
import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .features import FEATURE_NAMES


def load(path):
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    X = np.array([[float(r[n]) for n in FEATURE_NAMES] for r in rows])
    y = np.array([int(float(r["label"])) for r in rows])
    weighted = np.array([float(r["weighted_suspicion"]) for r in rows])
    caps = np.array([r["capture"] for r in rows])
    return X, y, weighted, caps


def metrics(score, y, detect_t, clear_t):
    det = score >= detect_t
    clr = score <= clear_t
    rev = ~det & ~clr
    tp = int(np.sum(det & (y == 1)))
    fp = int(np.sum(det & (y == 0)))
    fn = int(np.sum(~det & (y == 1)))          # review counts as not detected here
    tn = int(np.sum(~det & (y == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "roc_auc": round(float(roc_auc_score(y, score)), 4),
        "avg_precision": round(float(average_precision_score(y, score)), 4),
        "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
        "review_rate": round(float(rev.mean()), 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "missed_in_clear_band": int(np.sum(clr & (y == 1))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="models/learned_fusion.joblib")
    ap.add_argument("--out", default="outputs/review1/fusion_eval.json")
    args = ap.parse_args()

    X, y, weighted, caps = load(args.data)
    print(f"Rows: {len(y)} | positives: {int(y.sum())} | captures: {len(set(caps))}")
    if y.sum() == 0 or y.sum() == len(y):
        raise SystemExit("Need both positive and negative candidates to evaluate.")

    bundle = joblib.load(args.model)
    raw = bundle["model"].predict_proba(X)[:, 1]
    lg = np.log(np.clip(raw, 1e-4, 1 - 1e-4) / (1 - np.clip(raw, 1e-4, 1 - 1e-4))).reshape(-1, 1)
    learned = bundle["calibrator"].predict_proba(lg)[:, 1]
    bayes = X[:, FEATURE_NAMES.index("bayes_h0")]

    results = {
        "bayes_only": metrics(bayes, y, 0.6, 0.4),
        "weighted": metrics(weighted, y, 0.6, 0.4),
        "learned": metrics(learned, y, bundle["detect_threshold"], bundle["clear_threshold"]),
    }

    cols = ["roc_auc", "avg_precision", "precision", "recall", "f1", "review_rate", "missed_in_clear_band"]
    print(f"\n{'method':12s}" + "".join(f"{c:>24s}" for c in cols))
    for name, m in results.items():
        print(f"{name:12s}" + "".join(f"{m[c]:>24}" for c in cols))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()