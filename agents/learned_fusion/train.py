"""
train.py - train the learned fusion (stacked meta-classifier) on candidate rows.

    python -m agents.learned_fusion.train --data data/fusion/val.csv
Options: --algo auto|lr|gb   --folds 5   --out models/learned_fusion.joblib

Capture-grouped cross-validation is used, so candidates from the same capture never appear in
both the training and validation folds. Probabilities are Platt-calibrated on out-of-fold scores,
and the Detected / Review / Not-Detected thresholds are chosen from those calibrated scores.
"""
import argparse
import csv
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_NAMES


def load_csv(paths):
    X, y, groups = [], [], []
    for p in paths:
        with open(p, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                X.append([float(r[n]) for n in FEATURE_NAMES])
                y.append(int(float(r["label"])))
                groups.append(r["capture"])
    return np.array(X), np.array(y), np.array(groups)


def make_model(algo):
    if algo == "lr":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(class_weight="balanced", max_iter=3000, C=1.0))
    return GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                      subsample=0.8, random_state=42)


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p)).reshape(-1, 1)


def best_f1_threshold(p, y):
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.01):
        pred = p >= t
        tp = int(np.sum(pred & (y == 1)))
        fp = int(np.sum(pred & (y == 0)))
        fn = int(np.sum(~pred & (y == 1)))
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t, best_f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", default="models/learned_fusion.joblib")
    ap.add_argument("--algo", default="auto", choices=["auto", "lr", "gb"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max-missed", type=float, default=0.05,
                    help="share of true camouflage allowed below the 'not detected' threshold")
    args = ap.parse_args()

    X, y, groups = load_csv(args.data)
    print(f"Rows: {len(y)} | positives: {int(y.sum())} | negatives: {int((1 - y).sum())} "
          f"| captures: {len(set(groups))}")
    if y.sum() == 0 or y.sum() == len(y):
        raise SystemExit("Need both positive and negative candidates to train.")

    n_splits = max(2, min(args.folds, len(set(groups))))
    cv = list(GroupKFold(n_splits=n_splits).split(X, y, groups))

    algos = ["lr", "gb"] if args.algo == "auto" else [args.algo]
    results = {}
    for a in algos:
        oof = cross_val_predict(make_model(a), X, y, cv=cv, method="predict_proba")[:, 1]
        results[a] = {"oof": oof, "auc": roc_auc_score(y, oof), "ap": average_precision_score(y, oof)}
        print(f"  {a}: grouped-CV ROC-AUC {results[a]['auc']:.3f} | average precision {results[a]['ap']:.3f}")

    algo = max(results, key=lambda k: results[k]["ap"])
    print(f"Selected: {algo}")

    # Platt calibration on out-of-fold scores
    oof = results[algo]["oof"]
    calibrator = LogisticRegression(C=1e6, max_iter=1000).fit(logit(oof), y)
    cal_oof = calibrator.predict_proba(logit(oof))[:, 1]

    detect_t, f1 = best_f1_threshold(cal_oof, y)
    pos_scores = cal_oof[y == 1]
    clear_t = float(np.quantile(pos_scores, args.max_missed))
    clear_t = max(0.02, min(clear_t, detect_t - 0.05))
    print(f"Thresholds: detect >= {detect_t:.2f} (CV F1 {f1:.3f}) | not detected <= {clear_t:.2f}")

    model = make_model(algo).fit(X, y)

    if algo == "lr":
        coefs = model[-1].coef_[0]
        order = np.argsort(-np.abs(coefs))[:10]
        print("Top features (|scaled coefficient|):")
        for i in order:
            print(f"   {FEATURE_NAMES[i]:20s} {coefs[i]:+.3f}")
    else:
        imp = model.feature_importances_
        order = np.argsort(-imp)[:10]
        print("Top features (importance):")
        for i in order:
            print(f"   {FEATURE_NAMES[i]:20s} {imp[i]:.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model, "calibrator": calibrator, "algo": algo,
        "feature_names": FEATURE_NAMES,
        "detect_threshold": detect_t, "clear_threshold": clear_t,
        "cv": {"roc_auc": results[algo]["auc"], "average_precision": results[algo]["ap"], "f1": f1},
        "n_rows": int(len(y)), "n_positive": int(y.sum()),
    }, out)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()