"""Post-hoc alpha-threshold sweep over saved ist_explain results.

The NLL solver's false positives cluster at small alpha (tag-alongs just
above the solver's survival threshold) while true recoveries sit higher.
This script sweeps a minimum-alpha cutoff over an existing results JSON and
reports micro precision/recall/F1 per method — the operating curve for
choosing an explanation-level threshold, with no GPU re-run.

Usage:
    python scripts/ist_alpha_sweep.py results/ist_explain_vanilla_nll.json
"""

import argparse
import json


def micro(pairs, method, alpha_min, require_verified):
    tp = fp = fn = 0
    for p in pairs:
        if method not in p or "true_edits" not in p:
            continue
        sol = p[method]
        preds = {e["concept_id"] for e in sol["explanation"]
                 if e["alpha"] >= alpha_min}
        if require_verified and not sol.get("verified", True):
            preds = set()
        true = {e["concept_id"] for e in p["true_edits"]}
        tp += len(preds & true)
        fp += len(preds - true)
        fn += len(true - preds)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1, tp, fp, fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_file")
    parser.add_argument("--thresholds", type=float, nargs="+",
                        default=[0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5])
    parser.add_argument("--no-verified", action="store_true",
                        help="do not zero out dNLL-rejected explanations")
    args = parser.parse_args()

    data = json.load(open(args.results_file))
    pairs = data["pairs"]
    methods = [m for m in ("sparse", "greedy", "nll")
               if any(m in p for p in pairs)]

    for method in methods:
        print(f"\n=== {method} (dNLL verification "
              f"{'off' if args.no_verified else 'on'}) ===")
        print(f"{'alpha_min':>10} {'prec':>7} {'recall':>7} {'f1':>7} "
              f"{'tp':>4} {'fp':>4} {'fn':>4}")
        best = None
        for t in args.thresholds:
            prec, rec, f1, tp, fp, fn = micro(
                pairs, method, t, not args.no_verified)
            mark = ""
            if best is None or f1 > best[0]:
                best = (f1, t)
            print(f"{t:>10.3f} {prec:>7.3f} {rec:>7.3f} {f1:>7.3f} "
                  f"{tp:>4} {fp:>4} {fn:>4}")
        print(f"best F1 {best[0]:.3f} at alpha_min={best[1]}")


if __name__ == "__main__":
    main()
