"""
inspect_candidates.py - see every candidate's evidence and where the simple fusion puts it.

    python -m agents.fusion.inspect_candidates                 (top 15 by score)
    python -m agents.fusion.inspect_candidates --ids 3 7       (show candidates C3 and C7)
    python -m agents.fusion.inspect_candidates --detect 0.40 --clear 0.20
Reads outputs/review1/08_cognitive_results.json, written by the pipeline.
"""
import argparse
import json

from .simple_fusion import SimpleFusionAgent, SimpleFusionConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="outputs/review1/08_cognitive_results.json")
    ap.add_argument("--ids", nargs="*", type=int, default=None)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--detect", type=float, default=0.45)
    ap.add_argument("--clear", type=float, default=0.25)
    args = ap.parse_args()

    with open(args.file, encoding="utf-8") as f:
        cands = json.load(f)

    agent = SimpleFusionAgent(SimpleFusionConfig(detect_threshold=args.detect, clear_threshold=args.clear))
    rows = []
    for i, c in enumerate(cands, start=1):
        r = agent.fuse_candidate(c)
        rows.append((i, c, r))

    shown = [row for row in rows if args.ids is None or row[0] in args.ids]
    if args.ids is None:
        shown = sorted(shown, key=lambda t: -t[2]["final_suspicion"])[:args.top]

    counts = {}
    for _, _, r in rows:
        counts[r["final_status"]] = counts.get(r["final_status"], 0) + 1
    print(f"{len(rows)} candidates | {counts}\n")
    print(f"{'id':>4} {'status':<20} {'score':>6} {'spat':>6} {'spec':>6} {'feat':>6} {'agree':>6} "
          f"{'cog':>6} {'conf':>6} {'unc':>6}  bbox [x,y,w,h]")
    for i, c, r in shown:
        s = r["fusion_signals"]
        print(f"C{i:<3} {r['final_status']:<20} {r['final_suspicion']:>6.2f} {s['spatial']:>6.2f} "
              f"{s['spectral']:>6.2f} {s['feature']:>6.2f} {s['agreement']:>6.2f} {s['cognitive']:>6.2f} "
              f"{float(c.get('conflict', 0)):>6.2f} {float(c.get('uncertainty', 0)):>6.2f}  {c.get('bbox')}")


if __name__ == "__main__":
    main()