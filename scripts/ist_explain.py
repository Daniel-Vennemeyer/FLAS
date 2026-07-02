"""Explain response pairs as sparse concept transports (inverse problem).

For each pair (prompt, response_a, response_b), extracts layer-l activations
of both responses, then finds the sparse non-negative concept strengths whose
mixture transport best moves response_a's activation distribution onto
response_b's (flas.ist.inverse). Reports per-pair:

  - the explanation: [(concept, alpha), ...] sorted by strength
  - explained fraction: 1 - d(transport(h_a), h_b) / d(h_a, h_b)
  - behavioral delta-NLL: NLL of y_b under the frozen LM, unsteered minus
    steered by the inferred transport (positive = transport makes y_b likelier)
  - seed stability: mean pairwise Jaccard of supports across --seeds runs

If pairs carry ground-truth "edits" (from ist_make_synthetic_pairs.py), also
reports recovery precision/recall/F1 and the Spearman correlation between
inferred alphas and true strengths over recovered edits.

Usage:
    python scripts/ist_explain.py \
        --flow-ckpt checkpoints/ist_graded/best_step*.pt \
        --pairs-file data/ist_synthetic_pairs.json \
        --concept-bank data/ist_concept_bank.json \
        --method sparse --seeds 3 \
        --output results/ist_explain.json
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from flas.generate import load_generator
from flas.ist.activations import extract_layer_activations
from flas.ist.inverse import (
    TransportMixture, bank_displacements, shared_subspace, solve_greedy,
    solve_sparse, steered_nll, warm_start_alphas)


def load_bank(path):
    raw = json.load(open(path))
    bank = []
    for i, item in enumerate(raw):
        if isinstance(item, str):
            bank.append({"concept_id": i, "concept": item})
        else:
            bank.append({"concept_id": item.get("concept_id", i),
                         "concept": item["concept"]})
    return bank


def encode_bank(concept_enc, tokenizer, bank, max_len=64, device="cuda"):
    enc = tokenizer([b["concept"] for b in bank], return_tensors="pt",
                    padding=True, truncation=True, max_length=max_len).to(device)
    with torch.no_grad():
        hidden = concept_enc(enc.input_ids, enc.attention_mask)
    return hidden.float(), enc.attention_mask.float()


def spearman(x, y):
    if len(x) < 3:
        return None
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def recovery_metrics(pred_support, true_ids, bank):
    pred_ids = {bank[i]["concept_id"] for i in pred_support}
    true_ids = set(true_ids)
    tp = len(pred_ids & true_ids)
    precision = tp / len(pred_ids) if pred_ids else 0.0
    recall = tp / len(true_ids) if true_ids else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall > 0 else 0.0)
    return {"precision": precision, "recall": recall, "f1": f1,
            "true_positives": sorted(pred_ids & true_ids),
            "false_positives": sorted(pred_ids - true_ids),
            "missed": sorted(true_ids - pred_ids)}


def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow-ckpt", type=str, required=True)
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--num-blocks", type=int, default=None)
    parser.add_argument("--pairs-file", type=str, required=True)
    parser.add_argument("--concept-bank", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--method", choices=["sparse", "greedy", "both"],
                        default="sparse")
    parser.add_argument("--distance", choices=["proj", "mean_l2", "mmd"],
                        default="proj",
                        help="proj (default): penalize only the residual along "
                             "the a->b diff direction, discounting orthogonal "
                             "displacement by --orth-weight. mean_l2/mmd are "
                             "the strict distributional variants (mean_l2 "
                             "punishes the orthogonal bulk of steering "
                             "displacements and typically returns empty "
                             "explanations).")
    parser.add_argument("--orth-weight", type=float, default=0.1,
                        help="proj distance: weight on displacement orthogonal "
                             "to the a->b diff (1.0 ~= mean_l2)")
    parser.add_argument("--deflate-rank", type=int, default=1,
                        help="proj distance: project the rank-K shared "
                             "displacement subspace (bank mean + top K-1 PCs) "
                             "out of the metric, so concepts compete on "
                             "distinctive components only. 0 disables. Pick K "
                             "with the rank sweep in ist_diagnose.py.")
    parser.add_argument("--deflate-alpha", type=float, default=2.0,
                        help="reference strength for the shared subspace")
    parser.add_argument("--warm-start",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="initialize sparse alphas from per-concept "
                             "alignment scores instead of a uniform 0.1 "
                             "(uniform init superimposes the whole bank at "
                             "once and collapses to the empty solution)")
    parser.add_argument("--n-steps", type=int, default=3,
                        help="Euler steps of the transport mixture")
    parser.add_argument("--l1-weight", type=float, default=0.05)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--alpha-max", type=float, default=4.0)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--seeds", type=int, default=1,
                        help="Independent sparse solves for stability estimation")
    parser.add_argument("--concept-chunk", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--max-pairs", type=int, default=None)
    args = parser.parse_args()

    gen = load_generator(args.flow_ckpt, model_id=args.model_id,
                         layer=args.layer, num_blocks=args.num_blocks)
    llm, tokenizer, layer = gen.llm, gen.tokenizer, gen.layer
    prompt_format = getattr(gen, "_prompt_format", "chat")
    # Solve in fp32: gradients w.r.t. alpha through a bf16 flow are too noisy.
    flow_fn = gen.flow_fn.float()
    mixture = TransportMixture(flow_fn, n_steps=args.n_steps,
                               concept_chunk=args.concept_chunk)

    bank = load_bank(args.concept_bank)
    concept_hidden, concept_mask = encode_bank(gen.concept_enc, tokenizer, bank)
    print(f"Concept bank: {len(bank)} concepts")

    pairs = json.load(open(args.pairs_file))
    if args.max_pairs:
        pairs = pairs[:args.max_pairs]

    results = []
    for pair in pairs:
        pid = pair.get("pair_id", len(results))
        print(f"\npair {pid}: {pair['prompt'][:60]}...")
        h_a, mask_a, _ = extract_layer_activations(
            llm, tokenizer, layer, pair["prompt"], pair["response_a"],
            prompt_format, args.max_len)
        h_b, mask_b, _ = extract_layer_activations(
            llm, tokenizer, layer, pair["prompt"], pair["response_b"],
            prompt_format, args.max_len)

        entry = {"pair_id": pid, "prompt": pair["prompt"]}
        if "edits" in pair:
            entry["true_edits"] = pair["edits"]

        deflate, init_alphas = None, None
        if args.distance == "proj":
            disp = bank_displacements(
                mixture, h_a, mask_a, concept_hidden, concept_mask,
                alpha_ref=args.deflate_alpha)
            if args.deflate_rank > 0:
                deflate = shared_subspace(disp, rank=args.deflate_rank)
            if args.warm_start:
                init_alphas = warm_start_alphas(
                    disp, h_a, mask_a, h_b, mask_b, deflate=deflate)

        solutions = {}
        if args.method in ("sparse", "both"):
            per_seed = []
            for seed in range(args.seeds):
                res = solve_sparse(
                    mixture, h_a, mask_a, h_b, mask_b,
                    concept_hidden, concept_mask,
                    distance=args.distance, l1_weight=args.l1_weight,
                    iters=args.iters, lr=args.lr, alpha_max=args.alpha_max,
                    threshold=args.threshold, seed=seed,
                    orth_weight=args.orth_weight, deflate=deflate,
                    init_alphas=init_alphas)
                per_seed.append(res)
            canonical = per_seed[0]
            stability = None
            if len(per_seed) > 1:
                stability = float(np.mean([
                    jaccard(a.support, b.support)
                    for a, b in itertools.combinations(per_seed, 2)]))
            solutions["sparse"] = (canonical, stability)
        if args.method in ("greedy", "both"):
            res = solve_greedy(
                mixture, h_a, mask_a, h_b, mask_b,
                concept_hidden, concept_mask, distance=args.distance,
                orth_weight=args.orth_weight, deflate=deflate)
            solutions["greedy"] = (res, None)

        for name, (res, stability) in solutions.items():
            explanation = sorted(
                [{"concept_id": bank[i]["concept_id"],
                  "concept": bank[i]["concept"],
                  "alpha": round(float(res.alphas[i]), 3)}
                 for i in res.support],
                key=lambda e: -e["alpha"])

            nll_base = steered_nll(
                llm, tokenizer, layer, mixture, pair["prompt"],
                pair["response_b"], concept_hidden, concept_mask,
                torch.zeros_like(res.alphas), prompt_format, args.max_len)
            nll_steer = steered_nll(
                llm, tokenizer, layer, mixture, pair["prompt"],
                pair["response_b"], concept_hidden, concept_mask,
                res.alphas, prompt_format, args.max_len)

            sol = {
                "explanation": explanation,
                "explained_fraction": round(res.explained_fraction, 4),
                "nll_yb_unsteered": round(nll_base, 4),
                "nll_yb_steered": round(nll_steer, 4),
                "delta_nll": round(nll_base - nll_steer, 4),
            }
            if stability is not None:
                sol["seed_stability_jaccard"] = round(stability, 3)
            if "edits" in pair:
                sol["recovery"] = recovery_metrics(
                    res.support, [e["concept_id"] for e in pair["edits"]], bank)
            entry[name] = sol
            print(f"  [{name}] EF={sol['explained_fraction']:.3f} "
                  f"dNLL={sol['delta_nll']:+.3f} "
                  f"explanation={[(e['concept'][:30], e['alpha']) for e in explanation]}")
        results.append(entry)

    # Aggregate
    summary = {"n_pairs": len(results)}
    for name in ("sparse", "greedy"):
        rows = [r[name] for r in results if name in r]
        if not rows:
            continue
        agg = {
            "mean_explained_fraction": float(np.mean(
                [r["explained_fraction"] for r in rows])),
            "mean_delta_nll": float(np.mean([r["delta_nll"] for r in rows])),
            "mean_explanation_size": float(np.mean(
                [len(r["explanation"]) for r in rows])),
        }
        stab = [r["seed_stability_jaccard"] for r in rows
                if "seed_stability_jaccard" in r]
        if stab:
            agg["mean_seed_stability"] = float(np.mean(stab))
        recov = [r["recovery"] for r in rows if "recovery" in r]
        if recov:
            agg["recovery"] = {
                k: float(np.mean([r[k] for r in recov]))
                for k in ("precision", "recall", "f1")}
            # Strength calibration over recovered edits: inferred alpha vs
            # ground-truth ordinal strength.
            xs, ys = [], []
            for r in results:
                if name not in r or "true_edits" not in r:
                    continue
                true_strength = {e["concept_id"]: e["strength"]
                                 for e in r["true_edits"]}
                for e in r[name]["explanation"]:
                    if e["concept_id"] in true_strength:
                        xs.append(e["alpha"])
                        ys.append(true_strength[e["concept_id"]])
            rho = spearman(np.array(xs), np.array(ys)) if xs else None
            agg["strength_spearman"] = rho
            agg["n_recovered_edits"] = len(xs)
        summary[name] = agg

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"config": {k: v for k, v in vars(args).items()},
                   "summary": summary, "pairs": results}, f, indent=2)

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
