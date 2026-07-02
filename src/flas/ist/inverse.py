"""Inverse Semantic Transport: sparse mixture of concept flows.

Forward model (order-free joint Euler over M concepts with per-concept
strengths alpha_i, reducing exactly to single-concept FLAS when only one
alpha_i > 0):

    h_{k+1} = h_k + sum_i (alpha_i / N) * v_theta(h_k, k*alpha_i/N, c_i)

Inverse problem: given activations h_a, h_b of two responses to the same
prompt, find sparse non-negative strengths alpha minimizing

    D( pool(Phi_alpha(h_a)), pool(h_b) ) / D0  +  l1 * sum(alpha)

The whole pipeline is differentiable in alpha (through both the Euler step
size and the sinusoidal time embedding), so `solve_sparse` optimizes alpha
directly with Adam; `solve_greedy` is the grid-search baseline (Sec 8.1 of
the proposal). Run the FlowFunction in float32 for solver stability.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F

from flas.ist.activations import (
    activation_distance, encode_prompt_response, masked_mean)
from flas.model import get_text_decoder


class TransportMixture:
    """Applies the joint-Euler mixture of concept flows to an activation."""

    def __init__(self, flow_fn, n_steps=3, concept_chunk=8):
        self.flow_fn = flow_fn
        self.n_steps = n_steps
        self.concept_chunk = concept_chunk
        self._dtype = next(flow_fn.parameters()).dtype

    def transport(self, h, concept_hidden, concept_mask, alphas,
                  padding_mask=None):
        """Integrate the mixture flow.

        Args:
            h:              [1, S, d] activation (prompt + response positions)
            concept_hidden: [M, L, d] encoded concept bank
            concept_mask:   [M, L] float
            alphas:         [M] non-negative strengths (may require grad)
            padding_mask:   [1, S] float (default all-ones)
        Returns [1, S, d] transported activation.
        """
        assert h.size(0) == 1, "TransportMixture.transport expects batch size 1"
        h = h.to(self._dtype)
        concept_hidden = concept_hidden.to(self._dtype)
        n = self.n_steps
        m_total = alphas.numel()
        if padding_mask is None:
            padding_mask = torch.ones(1, h.size(1), device=h.device)
        padding_mask = padding_mask.float()

        for k in range(n):
            v_total = torch.zeros_like(h)
            for start in range(0, m_total, self.concept_chunk):
                sl = slice(start, min(start + self.concept_chunk, m_total))
                a = alphas[sl]
                m = a.numel()
                # Skip chunks that are exactly zero and detached (greedy path).
                if not a.requires_grad and torch.all(a == 0):
                    continue
                dt = (a / n).to(self._dtype)           # [m]
                t_k = dt.float() * k                   # [m], per-concept time
                h_rep = h.expand(m, -1, -1)
                v, _ = self.flow_fn(
                    h_rep, concept_hidden[sl], concept_mask[sl].float(),
                    t=t_k, padding_mask=padding_mask.expand(m, -1))
                v_total = v_total + (dt[:, None, None] * v).sum(dim=0, keepdim=True)
            h = h + v_total
        return h


@dataclass
class SolveResult:
    alphas: torch.Tensor              # [M] final strengths (thresholded)
    support: List[int]                # indices with alpha > 0
    d0: float                         # distance before transport
    d_final: float                    # distance after transport (final alphas)
    explained_fraction: float         # 1 - d_final / d0
    history: List[float] = field(default_factory=list)


def _distance_fn(kind, h_b, mask_b, h_a=None, mask_a=None, orth_weight=0.1):
    """Bind the target side; returns f(h, mask) -> scalar distance.

    kind="proj" decomposes the residual pool(h) - pool(h_b) into its component
    along u = unit(pool(h_b) - pool(h_a)) and the orthogonal remainder, and
    weights the orthogonal part by `orth_weight`:

        d = (<r, u>^2 + orth_weight * ||r - <r,u>u||^2) / dim

    Rationale: steering displacements are much larger than the pooled diff
    between two same-prompt responses, and mostly orthogonal to it (content
    vs. concept). Full L2 (orth_weight=1, equivalent to mean_l2) then punishes
    a *correct* concept for its orthogonal bulk; proj rewards movement toward
    h_b while only softly penalizing off-axis drift. Overshoot along u is
    still penalized, so alpha stays calibrated. d0 = ||diff||^2/dim either way,
    keeping the explained fraction comparable.
    """
    if kind == "proj":
        assert h_a is not None and mask_a is not None, "proj needs the h_a reference"
        pool_b = masked_mean(h_b, mask_b)              # [1, d]
        pool_a = masked_mean(h_a.float(), mask_a)
        diff = pool_b - pool_a
        u = diff / diff.norm().clamp(min=1e-8)
        dim = h_b.size(-1)

        def f(h, mask):
            r = masked_mean(h, mask) - pool_b
            along = (r * u).sum()
            orth = r - along * u
            return (along.pow(2) + orth_weight * orth.pow(2).sum()) / dim
        return f

    def f(h, mask):
        return activation_distance(h, mask, h_b, mask_b, kind=kind)
    return f


def _evaluate(mixture, h_a, mask_a, dist, alphas, concept_hidden,
              concept_mask, padding_mask):
    h_t = mixture.transport(h_a, concept_hidden, concept_mask, alphas,
                            padding_mask=padding_mask)
    return dist(h_t.float(), mask_a)


def solve_sparse(mixture, h_a, mask_a, h_b, mask_b,
                 concept_hidden, concept_mask, *,
                 distance="mean_l2", l1_weight=0.05, iters=200, lr=0.1,
                 alpha_max=4.0, alpha_init=0.1, threshold=0.05,
                 refit=True, seed=0, padding_mask=None, orth_weight=0.1,
                 verbose=False):
    """Gradient-based sparse inverse: alpha = alpha_max * sigmoid(rho),
    Adam on rho, L1 on alpha, hard-threshold + optional refit of survivors."""
    device = h_a.device
    m_total = concept_hidden.size(0)
    dist = _distance_fn(distance, h_b.float(), mask_b, h_a, mask_a, orth_weight)

    d0 = dist(h_a.float(), mask_a).item()
    if d0 < 1e-12:
        return SolveResult(torch.zeros(m_total, device=device), [], d0, d0, 0.0)

    gen = torch.Generator(device="cpu").manual_seed(seed)
    rho0 = math.log(alpha_init / (alpha_max - alpha_init))

    def _optimize(support_mask, n_iters):
        rho = torch.full((m_total,), rho0, device=device)
        rho += 0.01 * torch.randn(m_total, generator=gen).to(device)
        rho.requires_grad_(True)
        opt = torch.optim.Adam([rho], lr=lr)
        history = []
        for it in range(n_iters):
            opt.zero_grad()
            alphas = alpha_max * torch.sigmoid(rho) * support_mask
            d = _evaluate(mixture, h_a, mask_a, dist, alphas,
                          concept_hidden, concept_mask, padding_mask)
            loss = d / d0 + l1_weight * alphas.sum()
            loss.backward()
            opt.step()
            history.append(float(loss.detach()))
            if verbose and (it % 20 == 0 or it == n_iters - 1):
                print(f"    iter {it:4d}  d/d0={float(d.detach()) / d0:.4f}  "
                      f"|alpha|_1={float(alphas.detach().sum()):.3f}")
        with torch.no_grad():
            return alpha_max * torch.sigmoid(rho) * support_mask, history

    full_mask = torch.ones(m_total, device=device)
    alphas, history = _optimize(full_mask, iters)

    # Sigmoid never reaches exactly 0 — hard-threshold, then optionally refit
    # the surviving support so kept strengths are not biased low by the L1.
    keep = alphas >= threshold
    if refit and keep.any() and keep.sum() < m_total:
        alphas, hist2 = _optimize(keep.float(), max(iters // 2, 50))
        history += hist2
        keep = alphas >= threshold
    alphas = torch.where(keep, alphas, torch.zeros_like(alphas)).detach()

    with torch.no_grad():
        d_final = _evaluate(mixture, h_a, mask_a, dist, alphas,
                            concept_hidden, concept_mask, padding_mask).item()
    return SolveResult(
        alphas=alphas,
        support=sorted(torch.nonzero(alphas).flatten().tolist()),
        d0=d0, d_final=d_final,
        explained_fraction=1.0 - d_final / d0,
        history=history)


@torch.no_grad()
def solve_greedy(mixture, h_a, mask_a, h_b, mask_b,
                 concept_hidden, concept_mask, *,
                 distance="mean_l2", grid=(0.5, 1.0, 1.5, 2.0, 3.0),
                 max_k=4, min_rel_improve=0.02, padding_mask=None,
                 orth_weight=0.1, verbose=False):
    """Greedy bank search baseline: repeatedly add the (concept, strength)
    pair that most reduces the distance, until improvement saturates."""
    device = h_a.device
    m_total = concept_hidden.size(0)
    dist = _distance_fn(distance, h_b.float(), mask_b, h_a, mask_a, orth_weight)

    d0 = dist(h_a.float(), mask_a).item()
    if d0 < 1e-12:
        return SolveResult(torch.zeros(m_total, device=device), [], d0, d0, 0.0)

    alphas = torch.zeros(m_total, device=device)
    d_cur = d0
    for _round in range(max_k):
        best = None  # (d, idx, strength)
        for i in range(m_total):
            if alphas[i] > 0:
                continue
            for g in grid:
                cand = alphas.clone()
                cand[i] = g
                d = _evaluate(mixture, h_a, mask_a, dist, cand,
                              concept_hidden, concept_mask, padding_mask).item()
                if best is None or d < best[0]:
                    best = (d, i, g)
        if best is None or (d_cur - best[0]) / d0 < min_rel_improve:
            break
        d_cur, idx, g = best
        alphas[idx] = g
        if verbose:
            print(f"    greedy +concept[{idx}] @ {g}: d/d0={d_cur / d0:.4f}")

    return SolveResult(
        alphas=alphas,
        support=sorted(torch.nonzero(alphas).flatten().tolist()),
        d0=d0, d_final=d_cur,
        explained_fraction=1.0 - d_cur / d0)


@torch.no_grad()
def steered_nll(llm, tokenizer, layer, mixture, prompt, target,
                concept_hidden, concept_mask, alphas,
                prompt_format="chat", max_len=1024):
    """Teacher-forced NLL of `target` under the frozen LM with the mixture
    transport applied at `layer`. alphas of all-zeros gives the unsteered
    baseline; the gap is the behavioral-reconstruction metric."""
    device = next(llm.parameters()).device
    full_ids, prompt_len = encode_prompt_response(
        tokenizer, prompt, target, prompt_format, max_len)
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    labels = input_ids.clone()
    labels[0, :prompt_len] = -100

    apply_flow = bool(torch.any(alphas > 0))

    def hook(module, inputs, output):
        if not apply_flow:
            return output
        is_tuple = isinstance(output, tuple)
        h_orig = output[0] if is_tuple else output
        h = mixture.transport(
            h_orig.float(), concept_hidden, concept_mask, alphas)
        h_out = h.to(h_orig.dtype)
        return (h_out,) + output[1:] if is_tuple else h_out

    handle = get_text_decoder(llm).layers[layer].register_forward_hook(hook)
    try:
        out = llm(input_ids=input_ids, labels=labels)
    finally:
        handle.remove()
    return out.loss.item()
