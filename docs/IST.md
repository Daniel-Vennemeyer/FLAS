# Inverse Semantic Transport (IST)

IST extends FLAS from a steering method to an interpretability method. The
central claim under test: **language-model behavioral differences can be
explained as sparse compositions of natural-language concept transports in
activation space.** The forward model learns to move along concept dimensions
with calibrated intensity; the inverse model explains an observed response
difference by identifying which concept movements occurred — and validates
each explanation *causally*, by applying it back to the model.

Everything lives in `src/flas/ist/` (importable as `flas.ist`) plus four
`scripts/ist_*.py` drivers. FLAS itself (`flas.model`, `flas.train`,
`flas.generate`) is untouched.

## Components

### 1. Graded strength ↔ flow-time training (`flas.ist.train_graded`)

FLAS trains with `T ~ Uniform[0.5, 2]` and only *emergently* acquires
flow-time-as-strength. IST supervises it directly: each training sample
carries an ordinal strength `s ∈ {1, 2, 3}` (low/med/high) and teacher-forces
the response written at that strength with flow time `T = t_unit · s`
(multiplicative jitter `±t_jitter`). Supervision is purely behavioral
(LM loss) — there is **no activation-matching target**, because responses at
different strengths are different token sequences with no position-wise
activation correspondence.

```bash
uv run python -m flas.ist.train_graded \
    --data-dir data/graded \
    --t-unit 1.0 --t-jitter 0.25 --n-steps 3 \
    --output-dir checkpoints --run-name ist_graded
```

Checkpoints are standard FLAS checkpoints — `scripts/eval.py`, `chat.py`, and
the inverse solver all load them unchanged. A vanilla FLAS checkpoint also
works with the inverse solver (as the endpoint-only baseline).

### 2. Graded data construction (`scripts/ist_build_graded_data.py`)

For each (concept, prompt), one LLM call writes a neutral/low/medium/high
response ladder; a 0–10 strength judge then verifies the ladder is monotone
(rejecting inverted or flat ladders — LLM-written grading is noisy). Output
is a parquet with a `strength` column.

```bash
uv run python scripts/ist_build_graded_data.py \
    --concepts-file data/ist_concept_bank.json \
    --prompts-file data/alpaca_eval.json \
    --prompts-per-concept 8 --output-dir data/graded \
    --api-key $OPENAI_API_KEY
```

The concept bank is a JSON list of natural-language phrases (or
`{"concept_id": ..., "concept": ...}` objects), e.g. `"more cautious"`,
`"more emotionally validating"`, `"more concise"`.

### 3. The inverse solver (`flas.ist.inverse`)

Forward mixture model — order-free joint Euler over M concepts, reducing
exactly to single-concept FLAS when one strength is nonzero:

```
h_{k+1} = h_k + Σ_i (α_i / N) · v_θ(h_k, k·α_i/N, c_i)
```

Given activations `h_a`, `h_b` of two responses to the same prompt (captured
at the FLAS intervention layer), the solver finds sparse non-negative α
minimizing

```
D( pool(Φ_α(h_a)), pool(h_b) ) / D_0  +  λ · Σ α_i
```

- **Distances are distributional** (mean-pooled L2 or RBF-MMD over
  response-token point clouds) — never per-position, since `y_a` and `y_b`
  are different token sequences.
- **`solve_sparse`**: the whole pipeline is differentiable in α (through
  both the Euler step size and the sinusoidal time embedding), so α is
  optimized directly with Adam (`α = α_max · σ(ρ)`, L1 penalty,
  hard-threshold + refit). This is the primary method.
- **`solve_greedy`**: grid-search baseline — repeatedly add the
  (concept, strength) pair that most reduces the distance.
- **`solve_nll`** (`--method nll`): the token-level behavioral inverse —
  optimize α directly against the teacher-forced NLL of `y_b` under the
  frozen LM with the transport applied. Teacher forcing makes the objective
  per-position and alignment-free (no pooling, no cross-sequence matching),
  so it fully exercises the flow's per-token expressivity. Warm-started from
  the activation-space scores. Note: Δ-NLL verification is in-sample for
  this method — validate via ground-truth recovery and the causal eval.
- **`steered_nll`**: behavioral reconstruction/verification — NLL of `y_b`
  under the frozen LM with the inferred transport applied vs. unsteered.
  Explanations whose transport does not raise `y_b`'s likelihood by
  `--verify-dnll` are flagged REJECTED (activation-space matching proposes,
  behavioral reconstruction disposes).

Reported per pair: **explained fraction** `1 − D_final/D_0`, **Δ-NLL**, and
**seed stability** (Jaccard of supports across solver seeds — the diagnostic
for correlated bank entries like "more cautious" vs. "more hedging").

### 4. Synthetic-edit ground truth (`scripts/ist_make_synthetic_pairs.py`)

The flagship experiment. Construct pairs where `y_b` is `y_a` rewritten to
express K *known* concepts at *known* strengths (each edit verified by the
strength judge before the pair is accepted). Then score the solver on
recovering exactly those concepts:

```bash
uv run python scripts/ist_make_synthetic_pairs.py \
    --concepts-file data/ist_concept_bank.json \
    --num-pairs 100 --max-edits 3 \
    --output data/ist_synthetic_pairs.json --api-key $OPENAI_API_KEY

uv run python scripts/ist_explain.py \
    --flow-ckpt checkpoints/ist_graded/best_step*.pt \
    --pairs-file data/ist_synthetic_pairs.json \
    --concept-bank data/ist_concept_bank.json \
    --method both --seeds 3 \
    --output results/ist_explain.json
```

`ist_explain.py` reports recovery precision/recall/F1, the Spearman
correlation between inferred α and true ordinal strength (calibration), seed
stability, EF, and Δ-NLL. The bank passed to the solver should be a superset
of the edit concepts (distractors included) — recovery against distractors is
the point.

### 5. Causal validation (`scripts/ist_causal_eval.py`)

Apply each inferred explanation *forward*: generate fresh responses from the
same prompt with the inferred mixture (`MixtureFlasGenerator`, same joint-
Euler math as the solver), and judge per-concept strength of `y_a`, `y_b`,
and the steered generation. The explanation is causally valid if the steered
generation moves from `y_a`'s strength toward `y_b`'s:

```bash
uv run python scripts/ist_causal_eval.py \
    --flow-ckpt checkpoints/ist_graded/best_step*.pt \
    --explanations-file results/ist_explain.json \
    --output results/ist_causal.json --api-key $OPENAI_API_KEY
```

Reports movement agreement, normalized movement, and the α↔movement
correlation. This is the property descriptive baselines ("ask GPT how the
responses differ") cannot offer: their explanation cannot be applied back to
the model.

## Evaluation matrix (paper-1 scope)

| Axis | Metric | Where |
|---|---|---|
| Forward steering | AxBench C/I/F HMean (graded vs. vanilla FLAS ckpt) | existing `scripts/eval.py` |
| Strength calibration | judge strength vs. T monotonicity | `scripts/eval.py --flowtimes 0.5 1 2 3` + strength judge |
| Recovery | precision / recall / F1 vs. synthetic edits | `ist_explain.py` |
| Calibration | Spearman(α, true strength) | `ist_explain.py` |
| Stability | seed-Jaccard of supports | `ist_explain.py --seeds N` |
| Reconstruction | explained fraction, Δ-NLL of `y_b` | `ist_explain.py` |
| Causal validity | movement agreement, α↔movement corr. | `ist_causal_eval.py` |

Baselines to run against `solve_sparse`: `solve_greedy` (same transports),
vanilla FLAS checkpoint (endpoint-only flow, no graded supervision), and — 
external to this repo — DiffMean vector-arithmetic decomposition and an
LLM-describes-the-difference judge.

## Design decisions (and why)

- **No activation-matching loss.** `h^low` and `h^med` are activations of
  *different token sequences*; matching them per-position is ill-posed. All
  graded supervision is behavioral (LM loss at strength-mapped T), and all
  inverse objectives are distributional.
- **Strength-0 rows are dropped in training.** At T=0 the flow is the
  identity and contributes no gradient. Neutral responses exist only to
  anchor data construction and judging.
- **Order-free mixture, not composition.** Composed state-dependent flows
  don't commute; the joint-Euler mixture is order-free, differentiable, and
  reduces exactly to single-concept FLAS — and the same operator is used in
  the solver and in `MixtureFlasGenerator`, so inferred α transfer verbatim
  to generation.
- **No energy/naturalness regularizer, no ordinal probe loss (yet).** Only
  `h_N` enters the frozen model, so intermediate-state naturalness is not
  load-bearing for paper-1; a trained strength probe used as a loss invites
  Goodharting. Both are deferred as ablations — the 0–10 judge is used for
  *evaluation only*.

## Tests

CPU-only smoke tests (tiny random flow, no downloads) cover mixture identity
at α=0, chunking equivalence, gradient flow to α, both distances, and
end-to-end recovery of a planted sparse transport by both solvers:

```bash
uv run python tests/test_ist_smoke.py
```
