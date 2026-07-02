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
from flas.ist.inverse import TransportMixture, shared_subspace

DEFLATE_RANKS = (0, 1, 2, 3)


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
    parser.add_argument("--concept-chunk", type=int, default=24)
    parser.add_argument("--max-pairs", type=int, default=5)
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    gen = load_generator(args.flow_ckpt, model_id=args.model_id,
                         layer=args.layer, num_blocks=args.num_blocks)
    llm, tokenizer, layer = gen.llm, gen.tokenizer, gen.layer
    prompt_format = getattr(gen, "_prompt_format", "chat")
    flow_fn = gen.flow_fn.float()
    mixture = TransportMixture(flow_fn, n_steps=args.n_steps,
                               concept_chunk=args.concept_chunk)

    bank = load_bank(args.concept_bank)
    enc = tokenizer([b["concept"] for b in bank], return_tensors="pt",
                    padding=True, truncation=True, max_length=64).to("cuda")
    with torch.no_grad():
        concept_hidden = gen.concept_enc(enc.input_ids, enc.attention_mask).float()
    concept_mask = enc.attention_mask.float()

    pairs = json.load(open(args.pairs_file))[:args.max_pairs]
    all_rows = []
    ranks_by_r = {r: [] for r in DEFLATE_RANKS}

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
        disp_by_alpha = []
        with torch.no_grad():
            for alpha in args.alphas:
                states = mixture.individual_transports(
                    h_a, concept_hidden, concept_mask, alpha)
                m = mask_a.expand(states.size(0), -1)
                disp = masked_mean(states.float(), m) - pool_a.unsqueeze(0)  # [M, d]
                disp_by_alpha.append(disp)
                # Centered displacement: subtract the bank-mean (generic)
                # component so concepts are compared on what distinguishes them.
                cdisp = disp - disp.mean(dim=0, keepdim=True)
                for i, b in enumerate(bank):
                    d_i, cd_i = disp[i], cdisp[i]
                    pool_t = pool_a + d_i
                    rows.append({
                        "pair_id": pair.get("pair_id"),
                        "concept_id": b["concept_id"], "concept": b["concept"],
                        "is_true": b["concept_id"] in true_ids,
                        "alpha": alpha,
                        "cos": float(torch.nn.functional.cosine_similarity(
                            d_i, diff, dim=0)),
                        "ccos": float(torch.nn.functional.cosine_similarity(
                            cd_i, diff, dim=0)),
                        "ratio": float(d_i.norm() / diff.norm().clamp(min=1e-8)),
                        "proj_frac": float((d_i * u).sum()
                                           / diff.norm().clamp(min=1e-8)),
                        "ef_l2": float(1 - (pool_t - pool_b).pow(2).sum()
                                       / diff.pow(2).sum().clamp(min=1e-12)),
                    })
        all_rows.extend(rows)

        # Deflation-rank sweep: for each rank K, project the shared subspace
        # (bank mean + top K-1 PCs of the displacements) out of both sides and
        # rank concepts by best deflated cosine over the probed alphas. This
        # is exactly what the solver's --deflate-rank metric uses.
        with torch.no_grad():
            for r_k in DEFLATE_RANKS:
                best = None
                for disp in disp_by_alpha:
                    basis = shared_subspace(disp, rank=r_k)
                    if basis is None:
                        ddisp, ddiff = disp, diff
                    else:
                        ddisp = disp - (disp @ basis.T) @ basis
                        ddiff = (diff.unsqueeze(0)
                                 - (diff.unsqueeze(0) @ basis.T) @ basis)[0]
                    cs = torch.nn.functional.cosine_similarity(
                        ddisp, ddiff.unsqueeze(0), dim=1)  # [M]
                    best = cs if best is None else torch.maximum(best, cs)
                order = best.argsort(descending=True).tolist()
                rank_of = {bank[i]["concept_id"]: rank
                           for rank, i in enumerate(order, 1)}
                for cid in true_ids:
                    ranks_by_r[r_k].append(rank_of[cid])

        # Table: rank by best mean-centered cosine (ccos) for display.
        best_row = {}
        for r in rows:
            k = r["concept_id"]
            if k not in best_row or r["ccos"] > best_row[k]["ccos"]:
                best_row[k] = r
        ranked_c = sorted(best_row.values(), key=lambda r: -r["ccos"])

        print(f"{'rank':>4} {'ccos':>7} {'cos':>7} {'ratio':>7} {'proj':>7}  concept")
        for rank, r in enumerate(ranked_c, 1):
            if rank <= 8 or r["is_true"]:
                mark = " <-- TRUE" if r["is_true"] else ""
                extra = " (below top-8)" if rank > 8 else ""
                print(f"{rank:>4} {r['ccos']:>7.3f} {r['cos']:>7.3f} "
                      f"{r['ratio']:>7.2f} {r['proj_frac']:>7.3f}  "
                      f"{r['concept'][:45]}{mark}{extra}")

    n_true = len(ranks_by_r[0])
    m = len(bank)
    print(f"\n=== Signal summary over {len(pairs)} pairs, {n_true} true edits, "
          f"bank size {m} ===")
    if n_true:
        print("true-concept rank by deflated cosine (random mean = "
              f"{(m + 1) / 2:.1f}):")
        for r_k in DEFLATE_RANKS:
            tr = np.array(ranks_by_r[r_k])
            label = "raw (no deflation)" if r_k == 0 else f"deflate-rank {r_k}"
            print(f"  {label:<20} mean {tr.mean():>5.1f}  median "
                  f"{np.median(tr):>3.0f}  |  top-1 {np.mean(tr == 1):.2f}  "
                  f"top-3 {np.mean(tr <= 3):.2f}  top-5 {np.mean(tr <= 5):.2f}")
        print("(pass the best rank to ist_explain.py as --deflate-rank)")
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
