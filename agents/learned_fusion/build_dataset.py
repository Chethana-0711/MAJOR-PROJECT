"""
build_dataset.py - run every earlier layer on a split and save one row per candidate
(features from ALL layers + ground-truth label) for training / evaluating the learned fusion.

Run from the project root:
    python -m agents.learned_fusion.build_dataset --split val
    python -m agents.learned_fusion.build_dataset --split test
Options: --limit N (first N captures), --every K (every K-th capture), --resume
"""
import argparse
import csv
import json
from pathlib import Path

from agents.spatial_agent import SpatialAgent
from agents.spectral_agent import SpectralAgent
from agents.feature_agent import FeatureAgent
from agents.conflict_agent import ConflictAgent
from agents.uncertainty_agent import UncertaintyAgent
from agents.cognitive import CognitiveAgent as BayesCognitiveAgent, analyze_candidates
from agents.fusion import FusionAgent

from .features import FEATURE_NAMES, candidate_to_features
from .ground_truth import load_gt, label_candidate

PROJECT_DIR = Path(".")
SPLIT_FILE = PROJECT_DIR / "data" / "capture_splits.json"
MODEL_PATH = PROJECT_DIR / "runs" / "detect" / "train" / "weights" / "best.pt"

META = ["capture", "cand", "x", "y", "w", "h", "label", "best_iou", "mask_fraction",
        "weighted_suspicion", "weighted_status", "bayes_decision"]


def load_captures(split):
    with open(SPLIT_FILE, "r", encoding="utf-8") as f:
        entries = json.load(f)[split]
    captures = []
    for entry in entries:
        d = Path(str(entry).replace("\\", "/"))
        if not d.is_absolute():
            d = PROJECT_DIR / d
        if (d / "vis.png").exists() and (d / "lwir.png").exists() and (d / "label.png").exists():
            captures.append(d)
        else:
            print(f"WARNING: incomplete capture skipped: {d}")
    captures.sort(key=lambda p: str(p).lower())
    return captures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--mask-frac", type=float, default=0.6)
    args = ap.parse_args()

    out = Path(args.out) if args.out else PROJECT_DIR / "data" / "fusion" / f"{args.split}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    captures = load_captures(args.split)[::max(args.every, 1)]
    if args.limit:
        captures = captures[:args.limit]
    print(f"{args.split}: {len(captures)} captures -> {out}")

    done = set()
    if args.resume and out.exists():
        with open(out, newline="", encoding="utf-8") as f:
            done = {r["capture"] for r in csv.DictReader(f)}
        print(f"Resuming: {len(done)} captures already in file")

    print("Loading agents...")
    spatial = SpatialAgent(model_path=str(MODEL_PATH))
    spectral = SpectralAgent()
    feature = FeatureAgent()
    conflict = ConflictAgent()
    uncertainty = UncertaintyAgent()
    bayes = BayesCognitiveAgent()
    weighted = FusionAgent()          # baseline, rule signal absent so it matches the learned inputs

    mode = "a" if (args.resume and out.exists()) else "w"
    n_rows = n_pos = 0
    with open(out, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=META + FEATURE_NAMES)
        if mode == "w":
            writer.writeheader()

        for i, cap in enumerate(captures, start=1):
            if str(cap) in done:
                continue
            try:
                rgb, lwir = cap / "vis.png", cap / "lwir.png"
                s = spatial.analyze(rgb)
                sp = spectral.analyze(lwir)
                fe = feature.analyze(rgb_path=rgb, lwir_path=lwir)
                cf = conflict.analyze(s, sp, fe)
                un = uncertainty.analyze(cf)
                cands = analyze_candidates(bayes, un)
                gt_boxes, mask = load_gt(cap / "label.png")

                for k, c in enumerate(cands, start=1):
                    bbox = c.get("bbox")
                    if not bbox:
                        continue
                    c["rule_decision"] = "N/A"
                    base = weighted.fuse_candidate(c)
                    lab = label_candidate(bbox, gt_boxes, mask, args.iou, args.mask_frac)
                    row = {"capture": str(cap), "cand": k,
                           "x": bbox[0], "y": bbox[1], "w": bbox[2], "h": bbox[3],
                           "weighted_suspicion": base["final_suspicion"],
                           "weighted_status": base["final_status"],
                           "bayes_decision": c["bayes_decision"], **lab}
                    row.update(dict(zip(FEATURE_NAMES, candidate_to_features(c))))
                    writer.writerow(row)
                    n_rows += 1
                    n_pos += lab["label"]
                f.flush()
            except Exception as e:                      # keep going; report and continue
                print(f"  ERROR on {cap}: {e}")
            if i % 10 == 0 or i == len(captures):
                print(f"  {i}/{len(captures)} captures | rows {n_rows} | positives {n_pos}")

    print(f"Done. Saved {n_rows} candidate rows ({n_pos} positive) to {out}")


if __name__ == "__main__":
    main()