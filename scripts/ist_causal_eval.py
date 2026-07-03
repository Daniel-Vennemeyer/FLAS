"""Causal validation of inverse-transport explanations.

For each explained pair, applies the inferred concept mixture during fresh
generation from the same prompt (MixtureFlasGenerator) and judges, per
explained concept, the 0-10 strength of y_a, y_b, and the steered generation
y_hat. The explanation is causally valid for a concept if y_hat moves from
y_a's strength toward y_b's — i.e. the recovered transport, applied forward,
reproduces the behavioral difference it was inferred from. This is the
property that descriptive baselines ("ask an LLM how the responses differ")
cannot offer.

Reported per concept: movement agreement (sign of y_hat - y_a matches sign of
y_b - y_a) and normalized movement (y_hat - y_a) / (y_b - y_a). Aggregated
with the Pearson correlation between inferred alpha and achieved movement
(strength calibration).

Usage:
    python scripts/ist_causal_eval.py \
        --flow-ckpt checkpoints/ist_graded/best_step*.pt \
        --explanations-file results/ist_explain.json \
        --method sparse \
        --output results/ist_causal.json \
        --api-key $OPENAI_API_KEY
"""

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np

from flas.ist.mixture_generate import load_mixture_generator
from flas.ist.openai_utils import judge_strength, make_client


async def judge_all(client, model, jobs, concurrency=8):
    """jobs: list of (key, concept, text). Returns {key: score}."""
    sem = asyncio.Semaphore(concurrency)

    async def one(key, concept, text):
        async with sem:
            return key, await judge_strength(client, model, concept, text)

    out = await asyncio.gather(*[one(*j) for j in jobs])
    return dict(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow-ckpt", type=str, required=True)
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--num-blocks", type=int, default=None)
    parser.add_argument("--explanations-file", type=str, required=True,
                        help="Output of ist_explain.py")
    parser.add_argument("--method", choices=["sparse", "greedy", "nll"],
                        default="sparse")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--alpha-min", type=float, default=0.0,
                        help="drop explanation concepts below this strength "
                             "before generating/judging (trace-level alphas "
                             "have no judge-visible steering effect and "
                             "dilute movement agreement toward chance)")
    parser.add_argument("--alpha-scale", type=float, default=1.0,
                        help="multiply inferred alphas at generation time. "
                             "NLL-calibrated alphas can sit below the "
                             "strength needed for judge-visible effects in "
                             "free generation; if agreement holds and "
                             "movement grows with the scale, the inferred "
                             "strengths are relatively correct with a "
                             "different likelihood->generation gain")
    parser.add_argument("--n-steps", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--generations-per-pair", type=int, default=3,
                        help="Steered samples per pair (judged strengths averaged)")
    parser.add_argument("--judge-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--api-key", type=str, required=True)
    args = parser.parse_args()

    data = json.load(open(args.explanations_file))
    pairs_meta = {p["pair_id"]: p for p in json.load(
        open(data["config"]["pairs_file"]))}
    pairs = []
    for p in data["pairs"]:
        if args.method not in p:
            continue
        kept = [e for e in p[args.method]["explanation"]
                if e["alpha"] >= args.alpha_min]
        if kept:
            p = dict(p)
            p[args.method] = dict(p[args.method], explanation=kept)
            pairs.append(p)
    if args.max_pairs:
        pairs = pairs[:args.max_pairs]
    print(f"{len(pairs)} pairs with {args.method} explanations at "
          f"alpha >= {args.alpha_min} (generation alphas x{args.alpha_scale})")

    gen = load_mixture_generator(args.flow_ckpt, model_id=args.model_id,
                                 layer=args.layer, num_blocks=args.num_blocks)
    client = make_client(args.api_key)

    results = []
    for p in pairs:
        meta = pairs_meta[p["pair_id"]]
        explanation = p[args.method]["explanation"]
        concepts = [e["concept"] for e in explanation]
        alphas = [e["alpha"] * args.alpha_scale for e in explanation]

        outs = gen.generate_mixture(
            [meta["prompt"]] * args.generations_per_pair,
            concepts, alphas, n_steps=args.n_steps,
            max_tokens=args.max_tokens, temperature=args.temperature)
        y_hats = [o["generation"] for o in outs]

        jobs = []
        for ci, c in enumerate(concepts):
            jobs.append((("a", ci), c, meta["response_a"]))
            jobs.append((("b", ci), c, meta["response_b"]))
            for gi, y in enumerate(y_hats):
                jobs.append((("hat", ci, gi), c, y))
        scores = asyncio.run(judge_all(
            client, args.judge_model, jobs, args.concurrency))

        per_concept = []
        for ci, e in enumerate(explanation):
            s_a = scores.get(("a", ci))
            s_b = scores.get(("b", ci))
            s_hats = [scores.get(("hat", ci, gi))
                      for gi in range(len(y_hats))]
            s_hats = [s for s in s_hats if s is not None]
            if s_a is None or s_b is None or not s_hats:
                continue
            s_hat = float(np.mean(s_hats))
            target = s_b - s_a
            movement = s_hat - s_a
            per_concept.append({
                "concept": e["concept"], "alpha": e["alpha"],
                "strength_a": s_a, "strength_b": s_b,
                "strength_hat": s_hat,
                "movement": movement,
                "target_movement": target,
                "agrees": bool(movement * target > 0) if target != 0 else None,
                "normalized_movement": movement / target if target != 0 else None,
            })
        results.append({
            "pair_id": p["pair_id"],
            "generations": y_hats,
            "concepts": per_concept,
        })
        agr = [c["agrees"] for c in per_concept if c["agrees"] is not None]
        print(f"pair {p['pair_id']}: {sum(agr)}/{len(agr)} concepts moved "
              f"the right way" if agr else f"pair {p['pair_id']}: no judgeable concepts",
              flush=True)

    all_c = [c for r in results for c in r["concepts"]]
    agrees = [c["agrees"] for c in all_c if c["agrees"] is not None]
    norm = [c["normalized_movement"] for c in all_c
            if c["normalized_movement"] is not None]
    alphas_v = [c["alpha"] for c in all_c]
    movements = [c["movement"] for c in all_c]
    summary = {
        "n_pairs": len(results),
        "n_concept_evals": len(all_c),
        "movement_agreement": float(np.mean(agrees)) if agrees else None,
        "mean_normalized_movement": float(np.mean(norm)) if norm else None,
        "alpha_movement_pearson": (
            float(np.corrcoef(alphas_v, movements)[0, 1])
            if len(all_c) >= 3 and np.std(alphas_v) > 0 else None),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"config": {k: v for k, v in vars(args).items()
                              if k != "api_key"},
                   "summary": summary, "pairs": results}, f, indent=2)
    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
