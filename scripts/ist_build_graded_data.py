"""Build graded concept-strength training data for flas.ist.train_graded.

For each (concept, prompt) pair, one LLM call writes a 4-level ladder of
responses — neutral / low / medium / high concept expression — and a judge
pass verifies the ladder is actually monotone in concept strength (0-10
judge; ladders that invert or span too little are rejected, since LLM-written
"grading" is noisy). Accepted rows go to a parquet with a `strength` column
(0=neutral, 1=low, 2=medium, 3=high) compatible with flas.ist.train_graded.

Usage:
    python scripts/ist_build_graded_data.py \
        --concepts-file data/ist_concept_bank.json \
        --prompts-file data/alpaca_eval.json \
        --prompts-per-concept 8 \
        --output-dir data/graded \
        --api-key $OPENAI_API_KEY --concurrency 8
"""

import argparse
import asyncio
import json
import random
from pathlib import Path

import pandas as pd

from flas.ist.openai_utils import (
    call_with_backoff, judge_strength, make_client, parse_json_response)


LADDER_PROMPT = """You are constructing calibration data for controllable text \
generation. Given an instruction and a concept, write FOUR complete responses \
to the instruction that express the concept at increasing intensities:

- "neutral": a good response that does NOT express the concept at all.
- "low": the same kind of response with the concept expressed mildly, in one \
or two places.
- "medium": the concept clearly present through much of the response.
- "high": the concept strongly and pervasively expressed throughout.

All four responses must genuinely answer the instruction, be fluent, and be \
similar in length (80-150 words each). Change ONLY the degree of concept \
expression between levels; keep topic, structure, and quality as constant as \
possible.

Instruction: {instruction}

Concept: {concept}

Return a JSON object with exactly the keys "neutral", "low", "medium", "high", \
each mapping to the full response text."""

LEVELS = ["neutral", "low", "medium", "high"]


def load_concepts(path):
    """JSON list of {"concept_id": int, "concept": str} (or plain strings)."""
    raw = json.load(open(path))
    concepts = []
    for i, item in enumerate(raw):
        if isinstance(item, str):
            concepts.append({"concept_id": i, "concept": item})
        else:
            concepts.append({"concept_id": item.get("concept_id", i),
                             "concept": item["concept"]})
    return concepts


def load_prompts(path):
    raw = json.load(open(path))
    return [r["instruction"] if isinstance(r, dict) else r for r in raw]


def check_monotone(scores, min_gap, tol=0.0):
    """Ladder passes if judged strengths are non-decreasing (within tol) and
    high - neutral >= min_gap."""
    if any(s is None for s in scores):
        return False
    for a, b in zip(scores, scores[1:]):
        if b < a - tol:
            return False
    return scores[-1] - scores[0] >= min_gap


async def build_ladder(client, args, concept, prompt):
    completion = await call_with_backoff(
        client, args.gen_model,
        LADDER_PROMPT.format(instruction=prompt, concept=concept["concept"]),
        temperature=args.temperature, max_tokens=2048, json_mode=True)
    ladder = parse_json_response(completion)
    if ladder is None or any(lv not in ladder or not str(ladder[lv]).strip()
                             for lv in LEVELS):
        return None, "generation_failed"

    scores = await asyncio.gather(*[
        judge_strength(client, args.judge_model, concept["concept"],
                       str(ladder[lv]))
        for lv in LEVELS])
    if not check_monotone(list(scores), args.min_gap, args.monotone_tol):
        return None, f"non_monotone scores={list(scores)}"

    rows = []
    for s, lv in enumerate(LEVELS):
        rows.append({
            "input": prompt,
            "output": str(ladder[lv]),
            "output_concept": concept["concept"],
            "concept_id": concept["concept_id"],
            "strength": s,
            "judge_strength": scores[s],
        })
    return rows, None


async def run(args):
    client = make_client(args.api_key)
    concepts = load_concepts(args.concepts_file)
    prompts = load_prompts(args.prompts_file)
    rng = random.Random(args.seed)

    tasks = []
    for concept in concepts:
        for prompt in rng.sample(prompts, min(args.prompts_per_concept, len(prompts))):
            tasks.append((concept, prompt))
    print(f"{len(concepts)} concepts x {args.prompts_per_concept} prompts = "
          f"{len(tasks)} ladders")

    sem = asyncio.Semaphore(args.concurrency)
    accepted, rejected = [], []
    done = 0

    async def worker(concept, prompt):
        nonlocal done
        async with sem:
            rows, err = await build_ladder(client, args, concept, prompt)
        done += 1
        if rows is None:
            rejected.append({"concept_id": concept["concept_id"],
                             "concept": concept["concept"],
                             "prompt": prompt, "reason": err})
        else:
            accepted.extend(rows)
        if done % 25 == 0:
            print(f"  {done}/{len(tasks)} ladders "
                  f"({len(accepted) // 4} accepted, {len(rejected)} rejected)",
                  flush=True)

    await asyncio.gather(*[worker(c, p) for c, p in tasks])

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(accepted)
    df.to_parquet(out_dir / "train_data.parquet")
    with open(out_dir / "rejected.jsonl", "w") as f:
        for r in rejected:
            f.write(json.dumps(r) + "\n")
    with open(out_dir / "build_config.json", "w") as f:
        cfg = {k: v for k, v in vars(args).items() if k != "api_key"}
        json.dump(cfg, f, indent=2)

    n_ladders = len(accepted) // 4
    print(f"\nAccepted {n_ladders}/{len(tasks)} ladders "
          f"({len(df)} rows) -> {out_dir / 'train_data.parquet'}")
    print(f"Rejected {len(rejected)} -> {out_dir / 'rejected.jsonl'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concepts-file", type=str, required=True,
                        help="JSON list of concepts (strings or {concept_id, concept})")
    parser.add_argument("--prompts-file", type=str, default="data/alpaca_eval.json")
    parser.add_argument("--prompts-per-concept", type=int, default=8)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--gen-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--judge-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--min-gap", type=float, default=4.0,
                        help="Required judged-strength gap between high and neutral")
    parser.add_argument("--monotone-tol", type=float, default=0.5,
                        help="Allowed judged-strength inversion between adjacent levels")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--api-key", type=str, required=True)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
