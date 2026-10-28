"""
evaluate.py - Compare fusion methods on a held-out candidate CSV.

Usage:
    python -m agents.learned_fusion.evaluate --data data/fusion/test.csv
    python -m agents.learned_fusion.evaluate --data data/fusion/test.csv --model models/learned_fusion.joblib

Methods compared:
  bayes_only   : Bayesian cognitive layer P(H0)
  weighted     : Reliability-weighted fusion
  simple       : Simple rule-based fusion
  spatial_only : Spatial agent evidence
  learned      : Trained learned-fusion meta-classifier

Candidates classified as REVIEW count as not detected for precision,
recall and F1-score.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from ..fusion.simple_fusion import SimpleFusionAgent
from .dataset import build_matrix, load_dataset, row_to_candidate
from .features import FEATURE_NAMES
from .learned_fusion_agent import LearnedFusionAgent


def metrics(score, y, detect_t, clear_t):
    """Calculate candidate-level evaluation metrics."""

    det = score >= detect_t
    clr = score <= clear_t
    rev = ~det & ~clr

    tp = int(np.sum(det & (y == 1)))
    fp = int(np.sum(det & (y == 0)))
    fn = int(np.sum(~det & (y == 1)))
    tn = int(np.sum(~det & (y == 0)))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    # ROC-AUC and average precision require both classes.
    if len(np.unique(y)) == 2:
        roc_auc = float(roc_auc_score(y, score))
        avg_precision = float(average_precision_score(y, score))
    else:
        roc_auc = float("nan")
        avg_precision = float("nan")

    return {
        "roc_auc": round(roc_auc, 4),
        "avg_precision": round(avg_precision, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "review_rate": round(float(rev.mean()), 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "missed_in_clear_band": int(np.sum(clr & (y == 1))),
    }


def learned_scores(bundle, rows):
    """Calculate calibrated learned-fusion probabilities."""

    use_context = bundle.get(
        "use_context",
        bundle["feature_names"] != FEATURE_NAMES,
    )

    X, names = build_matrix(rows, use_context)

    if names != bundle["feature_names"]:
        raise SystemExit(
            "Model features do not match this CSV. "
            "Retrain with the current code or use the matching model."
        )

    raw = bundle["model"].predict_proba(X)[:, 1]
    raw = np.clip(raw, 1e-4, 1 - 1e-4)

    logits = np.log(raw / (1 - raw)).reshape(-1, 1)

    return bundle["calibrator"].predict_proba(logits)[:, 1]


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate and compare fusion methods."
    )

    parser.add_argument("--data", required=True)
    parser.add_argument(
        "--model",
        default="models/learned_fusion.joblib",
    )
    parser.add_argument(
        "--out",
        default="outputs/review1/fusion_eval.json",
    )

    args = parser.parse_args()

    # Load the held-out candidate dataset.
    _, y, captures, _, rows, _ = load_dataset(
        [args.data],
        use_context=False,
    )

    print(
        f"Rows: {len(y)} | "
        f"positives: {int(y.sum())} | "
        f"captures: {len(set(captures))}"
    )

    if len(y) == 0:
        raise SystemExit("The dataset contains no candidates.")

    if y.sum() == 0 or y.sum() == len(y):
        raise SystemExit(
            "Evaluation requires both positive and negative candidates."
        )

    # Load the trained learned-fusion model.
    bundle = joblib.load(args.model)
    learned = learned_scores(bundle, rows)

    # Obtain scores from the other fusion methods.
    weighted = np.array(
        [float(r["weighted_suspicion"]) for r in rows]
    )

    bayes = np.array(
        [float(r["bayes_h0"]) for r in rows]
    )

    spatial = np.array(
        [
            float(r["spatial_evidence"])
            * float(r["spatial_present"])
            for r in rows
        ]
    )

    simple_agent = SimpleFusionAgent()

    simple = np.array(
        [
            simple_agent.score(row_to_candidate(r))[0]
            for r in rows
        ]
    )

    sc = simple_agent.simple

    # Apply the same candidate-validation method used by the
    # learned fusion agent. Do not invent a second size rule.
    rejected = []

    for row in rows:
        candidate = row_to_candidate(row)

        reason = LearnedFusionAgent._reject_if_implausible(
            candidate
        )

        rejected.append(reason is not None)

    rejected = np.asarray(rejected, dtype=bool)

    rejected_positive = int(np.sum(rejected & (y == 1)))

    print(
        f"Rejected by candidate validation: {int(rejected.sum())} "
        f"candidates ({rejected_positive} positive targets)"
    )

    # Preserve the existing evaluation convention:
    # rejected candidates receive score zero.
    bayes, weighted, simple, spatial, learned = (
        np.where(rejected, 0.0, scores)
        for scores in (
            bayes,
            weighted,
            simple,
            spatial,
            learned,
        )
    )

    # Evaluate all methods using their existing thresholds.
    results = {
        "bayes_only": metrics(
            bayes, y, 0.6, 0.4
        ),
        "weighted": metrics(
            weighted, y, 0.6, 0.4
        ),
        "simple": metrics(
            simple,
            y,
            sc.detect_threshold,
            sc.clear_threshold,
        ),
        "spatial_only": metrics(
            spatial, y, 0.5, 0.25
        ),
        "learned": metrics(
            learned,
            y,
            bundle["detect_threshold"],
            bundle["clear_threshold"],
        ),
    }

    # Print a readable comparison table.
    columns = [
        "roc_auc",
        "avg_precision",
        "precision",
        "recall",
        "f1",
        "review_rate",
        "missed_in_clear_band",
    ]

    print(f"\n{'method':14s}", end="")

    for column in columns:
        print(f"{column:>22s}", end="")

    print()

    for name, result in results.items():
        print(
            f"{name:14s}"
            + "".join(
                f"{result[column]:>22}"
                for column in columns
            )
        )

    # Save results as JSON.
    output_path = Path(args.out)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(results, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()