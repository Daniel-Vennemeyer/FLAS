"""Build synthetic-edit response pairs with known ground-truth concepts.

The flagship IST evaluation: construct (y_a, y_b) pairs where y_b is y_a
rewritten to express K known concepts at known strengths. The inverse solver
is then scored on whether it recovers exactly those concepts (and calibrated
strengths) from activations alone — see scripts/ist_explain.py.

Each pair records its ground truth: {"edits": [{concept_id, concept,
strength}]} with strength in {1, 2, 3} = low/medium/high.

Usage:
    python scripts/ist_make_synthetic_pairs.py \
        --concepts-file data/ist_concept_bank.json \
        --prompts-file data/alpaca_eval.json \
        --num-pairs 100 --max-edits 3 \
        --output data/ist_synthetic_pairs.json \
        --api-key $OPENAI_API_KEY --concurrency 8
"""

import argparse
import asyncio
import json
import random
from pathlib import Path

from flas.ist.openai_utils import (
    call_with_backoff, judge_strength, make_client, parse_json_response)

STRENGTH_WORDS = {1: "mildly, in one or two places",
                  2: "clearly, through much of the response",
                  3: "strongly and pervasively, throughout"}

PAIR_PROMPT = """You are constructing evaluation data for a text-attribution \
study. First write a neutral, high-quality response to the instruction below \
(100-180 words). Then rewrite that response so that it additionally expresses \
each listed concept at the specified intensity, while keeping the topic, \
structure, length, and overall quality as close to the original as possible. \
Do not add any concept that is not listed.

Instruction: {instruction}

Concepts to add in the rewrite:
{edit_list}

Return a JSON object with exactly two keys: "base" (the neutral response) and \
"edited" (the rewritten response)."""


def load_concepts(path):
    raw = json.load(open(path))
    concepts = []
    for i, item in enumerate(raw):
        if isinstance(item, str):
            concepts.append({"concept_id": i, "concept": item})
        else:
            concepts.append({"concept_id": item.get("concept_id", i),
                             "concept": item["concept"]})
    return concepts


async def build_pair(client, args, pair_id, prompt, edits):
    edit_list = "\n".join(
        f"- {e['concept']} — expressed {STRENGTH_WORDS[e['strength']]}"
        for e in edits)
    completion = await call_with_backoff(
        client, args.gen_model,
        PAIR_PROMPT.format(instruction=prompt, edit_list=edit_list),
        temperature=args.temperature, max_tokens=2048, json_mode=True)
    obj = parse_json_response(completion)
    if obj is None or not str(obj.get("base", "")).strip() \
            or not str(obj.get("edited", "")).strip():
        return None

    base, edited = str(obj["base"]), str(obj["edited"])

    # Verify each intended edit actually landed (judge gap on edited vs base)
    # so ground truth labels are trustworthy.
    for e in edits:
        s_base, s_edit = await asyncio.gather(
            judge_strength(client, args.judge_model, e["concept"], base),
            judge_strength(client, args.judge_model, e["concept"], edited))
        if s_base is None or s_edit is None or s_edit - s_base < args.min_gap:
            return None
        e["judge_base"] = s_base
        e["judge_edited"] = s_edit

    return {"pair_id": pair_id, "prompt": prompt,
            "response_a": base, "response_b": edited, "edits": edits}


async def run(args):
    client = make_client(args.api_key)
    concepts = load_concepts(args.concepts_file)
    prompts = [r["instruction"] if isinstance(r, dict) else r
               for r in json.load(open(args.prompts_file))]
    rng = random.Random(args.seed)

    specs = []
    for pair_id in range(args.num_pairs):
        prompt = rng.choice(prompts)
        k = rng.randint(1, args.max_edits)
        chosen = rng.sample(concepts, k)
        edits = [{"concept_id": c["concept_id"], "concept": c["concept"],
                  "strength": rng.randint(1, 3)} for c in chosen]
        specs.append((pair_id, prompt, edits))

    sem = asyncio.Semaphore(args.concurrency)
    pairs = []
    done = 0

    async def worker(pair_id, prompt, edits):
        nonlocal done
        async with sem:
            pair = await build_pair(client, args, pair_id, prompt, edits)
        done += 1
        if pair is not None:
            pairs.append(pair)
        if done % 10 == 0:
            print(f"  {done}/{len(specs)} ({len(pairs)} accepted)", flush=True)

    await asyncio.gather(*[worker(*s) for s in specs])
    pairs.sort(key=lambda p: p["pair_id"])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(pairs, f, indent=2)
    print(f"\nAccepted {len(pairs)}/{len(specs)} pairs -> {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concepts-file", type=str, required=True)
    parser.add_argument("--prompts-file", type=str, default="data/alpaca_eval.json")
    parser.add_argument("--num-pairs", type=int, default=100)
    parser.add_argument("--max-edits", type=int, default=3)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--gen-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--judge-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--min-gap", type=float, default=3.0,
                        help="Required judged-strength gap (edited - base) per edit")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--api-key", type=str, required=True)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
