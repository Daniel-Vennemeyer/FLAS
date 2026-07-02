"""CPU smoke tests for flas.ist — no model downloads, no GPU.

Uses a tiny randomly-initialized Gemma2-config FlowFunction. The key
end-to-end check is self-consistency of the inverse problem: plant a known
sparse transport (h_b = Phi_{alpha*}(h_a) under the same flow), then verify
solve_sparse recovers the planted support and explains most of the distance.

Run:  uv run python tests/test_ist_smoke.py   (or pytest tests/)
"""

import torch
from transformers.models.gemma2 import Gemma2Config

from flas.model import FlowFunction
from flas.ist.activations import activation_distance, masked_mean
from flas.ist.inverse import TransportMixture, solve_greedy, solve_sparse

torch.manual_seed(0)

D, S, M, L, N = 64, 12, 6, 5, 2


def tiny_flow():
    config = Gemma2Config(
        vocab_size=256, hidden_size=D, intermediate_size=2 * D,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16)
    flow = FlowFunction(config, num_blocks=1, time_conditioned=True)
    # Zero-init time-MLP makes the flow time-invariant; perturb so t matters.
    with torch.no_grad():
        for p in flow.time_embed.mlp[-1].parameters():
            p.add_(0.01 * torch.randn_like(p))
    return flow.float().eval()


def make_problem(flow):
    mixture = TransportMixture(flow, n_steps=N, concept_chunk=8)
    h_a = torch.randn(1, S, D)
    mask = torch.zeros(1, S)
    mask[0, 4:] = 1.0  # positions 0-3 = "prompt", rest = "response"
    concept_hidden = torch.randn(M, L, D)
    concept_mask = torch.ones(M, L)
    alphas_true = torch.zeros(M)
    alphas_true[1], alphas_true[4] = 2.0, 1.0
    with torch.no_grad():
        h_b = mixture.transport(h_a, concept_hidden, concept_mask, alphas_true)
    return mixture, h_a, mask, h_b, concept_hidden, concept_mask, alphas_true


def test_transport_shapes_and_identity():
    flow = tiny_flow()
    mixture = TransportMixture(flow, n_steps=N)
    h = torch.randn(1, S, D)
    ch, cm = torch.randn(M, L, D), torch.ones(M, L)
    out = mixture.transport(h, ch, cm, torch.zeros(M))
    assert out.shape == (1, S, D)
    assert torch.allclose(out, h, atol=1e-5), "alpha=0 must be the identity"
    out2 = mixture.transport(h, ch, cm, torch.tensor([1.0] + [0.0] * (M - 1)))
    assert (out2 - h).abs().max() > 1e-4, "nonzero alpha must move h"


def test_chunking_equivalence():
    flow = tiny_flow()
    h = torch.randn(1, S, D)
    ch, cm = torch.randn(M, L, D), torch.ones(M, L)
    alphas = torch.rand(M)
    out_big = TransportMixture(flow, n_steps=N, concept_chunk=8).transport(
        h, ch, cm, alphas)
    out_small = TransportMixture(flow, n_steps=N, concept_chunk=2).transport(
        h, ch, cm, alphas)
    assert torch.allclose(out_big, out_small, atol=1e-5), \
        "concept_chunk must not change the result"


def test_gradient_flows_to_alphas():
    flow = tiny_flow()
    for p in flow.parameters():
        p.requires_grad_(False)
    mixture = TransportMixture(flow, n_steps=N)
    h = torch.randn(1, S, D)
    ch, cm = torch.randn(M, L, D), torch.ones(M, L)
    alphas = torch.full((M,), 0.5, requires_grad=True)
    out = mixture.transport(h, ch, cm, alphas)
    out.pow(2).mean().backward()
    assert alphas.grad is not None and alphas.grad.abs().sum() > 0


def test_distances():
    h1, h2 = torch.randn(1, S, D), torch.randn(1, S, D)
    m = torch.ones(1, S)
    assert activation_distance(h1, m, h1, m, "mean_l2") < 1e-10
    assert activation_distance(h1, m, h2, m, "mean_l2") > 0
    assert activation_distance(h1, m, h1, m, "mmd") < 1e-5
    assert activation_distance(h1, m, h2, m, "mmd") > 0
    pooled = masked_mean(h1, m)
    assert pooled.shape == (1, D)


def test_sparse_recovery():
    flow = tiny_flow()
    for p in flow.parameters():
        p.requires_grad_(False)
    mixture, h_a, mask, h_b, ch, cm, alphas_true = make_problem(flow)

    res = solve_sparse(
        mixture, h_a, mask, h_b, mask, ch, cm,
        distance="mean_l2", l1_weight=0.01, iters=250, lr=0.15,
        alpha_max=4.0, threshold=0.1, seed=0)

    print(f"  true alpha: {alphas_true.tolist()}")
    print(f"  recovered:  {[round(a, 2) for a in res.alphas.tolist()]}")
    print(f"  EF={res.explained_fraction:.3f}  support={res.support}")

    assert res.explained_fraction > 0.5, \
        f"solver should explain most of the planted transport (EF={res.explained_fraction:.3f})"
    true_support = set(torch.nonzero(alphas_true).flatten().tolist())
    top2 = set(torch.topk(res.alphas, 2).indices.tolist())
    assert top2 == true_support, \
        f"top-2 recovered concepts {top2} != planted {true_support}"


def test_greedy_recovery():
    flow = tiny_flow()
    mixture, h_a, mask, h_b, ch, cm, alphas_true = make_problem(flow)
    res = solve_greedy(
        mixture, h_a, mask, h_b, mask, ch, cm,
        distance="mean_l2", grid=(0.5, 1.0, 2.0), max_k=3)
    print(f"  greedy EF={res.explained_fraction:.3f}  support={res.support}")
    assert res.explained_fraction > 0.4
    true_support = set(torch.nonzero(alphas_true).flatten().tolist())
    assert set(res.support) & true_support, \
        "greedy should find at least one planted concept"


if __name__ == "__main__":
    for fn in [test_transport_shapes_and_identity, test_chunking_equivalence,
               test_gradient_flows_to_alphas, test_distances,
               test_sparse_recovery, test_greedy_recovery]:
        print(f"{fn.__name__} ...")
        fn()
        print(f"{fn.__name__} PASSED\n")
    print("All IST smoke tests passed.")
