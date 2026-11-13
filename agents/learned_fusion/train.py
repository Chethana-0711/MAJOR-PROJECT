
"""
train.py - train the learned fusion (stacked meta-classifier) on candidate rows.

Example:
    python -m agents.learned_fusion.train --data data/fusion/val.csv --seed 42

Reproducibility:
    Run with --seed 42, --seed 123, and --seed 2026.
    Keep the validation and test datasets unchanged.

Options:
    --algo auto|lr|gb|rf|et|hgb|ens
    --folds 5
    --beta 1.0
    --out models/learned_fusion.joblib
    --seed 42

Training:
    Train on candidates from the VAL split.
    Keep the TEST csv exclusively for evaluate.py.
"""

import argparse
from pathlib import Path

import joblib
import numpy as np

from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier
)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .dataset import load_dataset


# --------------------------------------------------
# MODEL DEFINITIONS
# --------------------------------------------------

def _et(seed):
    return ExtraTreesClassifier(
        n_estimators=400,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed
    )


def _rf(seed):
    return RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed
    )


def _hgb(seed):
    return HistGradientBoostingClassifier(
        max_depth=4,
        learning_rate=0.05,
        max_iter=300,
        l2_regularization=1.0,
        random_state=seed
    )


def make_model(algo, seed=42):

    if algo == "lr":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=5000,
                C=0.3,
                random_state=seed
            )
        )

    if algo == "gb":
        return GradientBoostingClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            random_state=seed
        )

    if algo == "rf":
        return _rf(seed)

    if algo == "et":
        return _et(seed)

    if algo == "hgb":
        return _hgb(seed)

    if algo == "ens":
        return VotingClassifier(
            [
                ("et", _et(seed)),
                ("rf", _rf(seed)),
                ("hgb", _hgb(seed))
            ],
            voting="soft"
        )

    raise ValueError(algo)


# --------------------------------------------------
# CALIBRATION AND THRESHOLD FUNCTIONS
# --------------------------------------------------

def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p)).reshape(-1, 1)


def best_fbeta_threshold(p, y, beta=1.0):

    best_t, best_f = 0.5, -1.0
    b2 = beta * beta

    for t in np.arange(0.02, 0.98, 0.01):

        pred = p >= t

        tp = int(np.sum(pred & (y == 1)))
        fp = int(np.sum(pred & (y == 0)))
        fn = int(np.sum(~pred & (y == 1)))

        denom = (1 + b2) * tp + b2 * fn + fp

        f = (1 + b2) * tp / denom if denom else 0.0

        if f > best_f:
            best_t, best_f = float(t), f

    return best_t, best_f


# --------------------------------------------------
# MAIN TRAINING FUNCTION
# --------------------------------------------------

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--data", nargs="+", required=True)

    ap.add_argument(
        "--out",
        default="models/learned_fusion.joblib"
    )

    ap.add_argument(
        "--algo",
        default="auto",
        choices=["auto", "lr", "gb", "rf", "et", "hgb", "ens"]
    )

    ap.add_argument("--folds", type=int, default=5)

    ap.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="F-beta used to select the detection threshold"
    )

    ap.add_argument(
        "--max-missed",
        type=float,
        default=0.05,
        help="Share of positive targets allowed below the clear threshold"
    )

    ap.add_argument(
        "--no-context",
        action="store_true",
        help="Use only the base features"
    )

    # NEW: configurable random seed
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible model training"
    )

    args = ap.parse_args()

    print("\n========================================")
    print("LEARNED FUSION TRAINING")
    print("========================================")
    print(f"Random seed: {args.seed}")
    print(f"Algorithm: {args.algo}")
    print(f"Folds: {args.folds}")

    # --------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------

    use_context = not args.no_context

    X, y, captures, scenes, _, names = load_dataset(
        args.data,
        use_context
    )

    print(
        f"Rows: {len(y)} | "
        f"positives: {int(y.sum())} | "
        f"negatives: {int((1 - y).sum())} | "
        f"captures: {len(set(captures))} | "
        f"scenes: {len(set(scenes))} | "
        f"features: {len(names)}"
    )

    if y.sum() == 0 or y.sum() == len(y):
        raise SystemExit(
            "Need both positive and negative candidates to train."
        )

    if y.sum() < 30:
        print(
            f"WARNING: only {int(y.sum())} positive examples - "
            "the model may be weak. Build the CSV from ALL val captures."
        )

    # --------------------------------------------------
    # SCENE-GROUPED CROSS VALIDATION
    # --------------------------------------------------

    n_splits = max(
        2,
        min(args.folds, len(set(scenes)))
    )

    cv = list(
        GroupKFold(n_splits=n_splits).split(X, y, scenes)
    )

    algos = (
        ["lr", "hgb", "ens"]
        if args.algo == "auto"
        else [args.algo]
    )

    results = {}

    print("\nCross-validation results:")

    for a in algos:

        oof = cross_val_predict(
            make_model(a, args.seed),
            X,
            y,
            cv=cv,
            method="predict_proba"
        )[:, 1]

        results[a] = {
            "oof": oof,
            "auc": roc_auc_score(y, oof),
            "ap": average_precision_score(y, oof)
        }

        print(
            f"  {a:4s}: "
            f"ROC-AUC {results[a]['auc']:.3f} | "
            f"Average Precision {results[a]['ap']:.3f}"
        )

    # --------------------------------------------------
    # SELECT BEST ALGORITHM
    # --------------------------------------------------

    algo = max(
        results,
        key=lambda k: results[k]["ap"]
    )

    print(f"\nSelected algorithm: {algo}")

    # --------------------------------------------------
    # PLATT CALIBRATION
    # --------------------------------------------------

    oof = results[algo]["oof"]

    calibrator = LogisticRegression(
        C=1e6,
        max_iter=1000,
        random_state=args.seed
    ).fit(
        logit(oof),
        y
    )

    cal_oof = calibrator.predict_proba(
        logit(oof)
    )[:, 1]

    # --------------------------------------------------
    # DETECTION THRESHOLDS
    # --------------------------------------------------

    detect_t, f = best_fbeta_threshold(
        cal_oof,
        y,
        args.beta
    )

    pos_scores = cal_oof[y == 1]

    clear_t = float(
        np.quantile(pos_scores, args.max_missed)
    )

    clear_t = max(
        0.02,
        min(clear_t, detect_t - 0.05)
    )

    # --------------------------------------------------
    # OUT-OF-FOLD METRICS
    # --------------------------------------------------

    pred = cal_oof >= detect_t

    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))

    precision = (
        tp / (tp + fp)
        if tp + fp
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    print("\nThresholds:")
    print(f"Detected threshold: {detect_t:.2f}")
    print(f"Clear threshold: {clear_t:.2f}")

    print("\nOut-of-fold metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    # --------------------------------------------------
    # FINAL MODEL TRAINING
    # --------------------------------------------------

    model = make_model(
        algo,
        args.seed
    ).fit(X, y)

    # --------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------

    if algo == "lr":

        coefs = model[-1].coef_[0]

        order = np.argsort(
            -np.abs(coefs)
        )[:10]

        print("\nTop features (|scaled coefficient|):")

        for i in order:
            print(
                f"   {names[i]:28s} {coefs[i]:+.3f}"
            )

    elif hasattr(model, "feature_importances_"):

        imp = model.feature_importances_

        order = np.argsort(-imp)[:10]

        print("\nTop features (importance):")

        for i in order:
            print(
                f"   {names[i]:28s} {imp[i]:.3f}"
            )

    # --------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------

    out = Path(args.out)

    out.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        {
            "model": model,
            "calibrator": calibrator,
            "algo": algo,
            "feature_names": names,
            "use_context": use_context,
            "detect_threshold": detect_t,
            "clear_threshold": clear_t,

            "cv": {
                "roc_auc": results[algo]["auc"],
                "average_precision": results[algo]["ap"],
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "beta": args.beta,
                "grouping": "scene"
            },

            "n_rows": int(len(y)),
            "n_positive": int(y.sum()),

            # NEW: save seed for experiment tracking
            "seed": args.seed
        },
        out
    )

    print("\n========================================")
    print("TRAINING COMPLETED")
    print("========================================")
    print(f"Algorithm: {algo}")
    print(f"Seed: {args.seed}")
    print(f"Saved model: {out}")


if __name__ == "__main__":
    main()
