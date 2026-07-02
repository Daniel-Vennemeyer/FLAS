"""Diagnose inverse-transport signal strength on ground-truth pairs.

For each pair and each bank concept c at each probe strength alpha, applies
the single-concept transport to h_a and measures the pooled displacement
disp = pool(Phi_alpha^c(h_a)) - pool(h_a) against the target diff
= pool(h_b) - pool(h_a):

    cos        cosine(disp, diff)      — is the concept moving the right way?
    ratio      ||disp|| / ||diff||     — how large is steering vs. the gap?
    proj_frac  <disp, u> / ||diff||    — fraction of the gap covered along it
    ef_l2      1 - ||pool_t - pool_b||^2 / ||diff||^2 (mean_l2 explained frac)

If true-edit concepts rank top by cos while ef_l2 stays <= 0, the solver's
failure is the objective (orthogonal displacement bulk swamping the aligned
component — use --distance proj in ist_explain.py), not a missing signal.
If true-edit concepts rank randomly by cos, the endpoint-only flow does not
encode recoverable concept directions at this layer.

Usage:
    python scripts/ist_diagnose.py \
        --flow-ckpt checkpoints/flas-gemma-2-2b-it/flas-gemma-2-2b-it.safetensors \
        --pairs-file data/ist_synthetic_pairs.json \
        --concept-bank data/ist_concept_bank.json \
        --max-pairs 5
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flas.generate import load_generator
from flas.ist.activations import extract_layer_activations, masked_mean
from flas.ist.inverse import TransportMixture


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow-ckpt", type=str, required=True)
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--num-blocks", type=int, default=None)
    parser.add_argument("--pairs-file", type=str, required=True)
    parser.add_argument("--concept-bank", type=str, required=True)
    parser.add_argument("--alphas", type=float, nargs="+", default=[1.0, 2.0, 3.0])
    parser.add_argument("--n-steps", type=int, default=3)
    parser.add_argument("--max-pairs", type=int, default=5)
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    gen = load_generator(args.flow_ckpt, model_id=args.model_id,
                         layer=args.layer, num_blocks=args.num_blocks)
    llm, tokenizer, layer = gen.llm, gen.tokenizer, gen.layer
    prompt_format = getattr(gen, "_prompt_format", "chat")
    flow_fn = gen.flow_fn.float()
    mixture = TransportMixture(flow_fn, n_steps=args.n_steps)

    bank = load_bank(args.concept_bank)
    enc = tokenizer([b["concept"] for b in bank], return_tensors="pt",
                    padding=True, truncation=True, max_length=64).to("cuda")
    with torch.no_grad():
        concept_hidden = gen.concept_enc(enc.input_ids, enc.attention_mask).float()
    concept_mask = enc.attention_mask.float()

    pairs = json.load(open(args.pairs_file))[:args.max_pairs]
    all_rows = []
    true_ranks = []

    for pair in pairs:
        h_a, mask_a, _ = extract_layer_activations(
            llm, tokenizer, layer, pair["prompt"], pair["response_a"],
            prompt_format, args.max_len)
        h_b, mask_b, _ = extract_layer_activations(
            llm, tokenizer, layer, pair["prompt"], pair["response_b"],
            prompt_format, args.max_len)
        pool_a = masked_mean(h_a, mask_a)[0]
        pool_b = masked_mean(h_b, mask_b)[0]
        diff = pool_b - pool_a
        u = diff / diff.norm().clamp(min=1e-8)
        true_ids = {e["concept_id"] for e in pair.get("edits", [])}

        print(f"\n=== pair {pair.get('pair_id')} ===")
        print(f"prompt: {pair['prompt'][:70]}")
        print(f"true edits: {[(e['concept'], e['strength']) for e in pair.get('edits', [])]}")
        print(f"||pool_a||={pool_a.norm():.2f}  ||diff||={diff.norm():.4f}  "
              f"(diff/pool_a = {diff.norm() / pool_a.norm():.4f})")

        rows = []
        with torch.no_grad():
            for i, b in enumerate(bank):
                for alpha in args.alphas:
                    a_vec = torch.zeros(1, device=h_a.device)
                    a_vec[0] = alpha
                    h_t = mixture.transport(
                        h_a, concept_hidden[i:i + 1], concept_mask[i:i + 1], a_vec)
                    pool_t = masked_mean(h_t.float(), mask_a)[0]
                    disp = pool_t - pool_a
                    cos = float(torch.nn.functional.cosine_similarity(
                        disp, diff, dim=0))
                    ratio = float(disp.norm() / diff.norm().clamp(min=1e-8))
                    proj_frac = float((disp * u).sum() / diff.norm().clamp(min=1e-8))
                    ef_l2 = float(1 - (pool_t - pool_b).pow(2).sum()
                                  / diff.pow(2).sum().clamp(min=1e-12))
                    rows.append({
                        "pair_id": pair.get("pair_id"),
                        "concept_id": b["concept_id"], "concept": b["concept"],
                        "is_true": b["concept_id"] in true_ids,
                        "alpha": alpha, "cos": cos, "ratio": ratio,
                        "proj_frac": proj_frac, "ef_l2": ef_l2,
                    })
        all_rows.extend(rows)

        # Rank concepts by their best cosine over the probed alphas.
        best_by_concept = {}
        for r in rows:
            k = r["concept_id"]
            if k not in best_by_concept or r["cos"] > best_by_concept[k]["cos"]:
                best_by_concept[k] = r
        ranked = sorted(best_by_concept.values(), key=lambda r: -r["cos"])
        print(f"{'rank':>4} {'cos':>7} {'ratio':>7} {'proj':>7} {'ef_l2':>7}  concept")
        for rank, r in enumerate(ranked[:8], 1):
            mark = " <-- TRUE" if r["is_true"] else ""
            print(f"{rank:>4} {r['cos']:>7.3f} {r['ratio']:>7.2f} "
                  f"{r['proj_frac']:>7.3f} {r['ef_l2']:>7.3f}  "
                  f"{r['concept'][:45]}{mark}")
        for rank, r in enumerate(ranked, 1):
            if r["is_true"]:
                true_ranks.append(rank)
                if rank > 8:
                    print(f"{rank:>4} {r['cos']:>7.3f} {r['ratio']:>7.2f} "
                          f"{r['proj_frac']:>7.3f} {r['ef_l2']:>7.3f}  "
                          f"{r['concept'][:45]} <-- TRUE (below top-8)")

    n_true = len(true_ranks)
    m = len(bank)
    print(f"\n=== Signal summary over {len(pairs)} pairs, {n_true} true edits, "
          f"bank size {m} ===")
    if n_true:
        tr = np.array(true_ranks)
        print(f"true-concept rank by cosine: mean {tr.mean():.1f} "
              f"(random = {(m + 1) / 2:.1f}), median {np.median(tr):.0f}")
        print(f"top-1 hit rate {np.mean(tr == 1):.2f}   "
              f"top-3 {np.mean(tr <= 3):.2f}   top-5 {np.mean(tr <= 5):.2f}")
    ratios = np.array([r["ratio"] for r in all_rows])
    print(f"displacement/diff ratio: median {np.median(ratios):.2f} "
          f"(>>1 means steering moves far beyond the response gap — "
          f"mean_l2 will punish it; use --distance proj)")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(all_rows, open(out, "w"), indent=2)
        print(f"per-(concept, alpha) rows -> {out}")


if __name__ == "__main__":
    main()
