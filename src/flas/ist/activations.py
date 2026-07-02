"""Activation extraction and distribution distances for inverse transport.

A (prompt, response) pair is teacher-forced through the frozen LM and the
output of decoder layer `layer` is captured — the same tensor the FLAS hook
intervenes on. Because y_a and y_b are different token sequences, there is no
position-wise correspondence between their activations; all comparisons here
are distributional over the response-token point clouds (mean-pooled L2 or
RBF-MMD), never per-position.
"""

import torch

from flas.model import get_text_decoder, build_chat_prompt

ALPACA_TEMPLATE = "### Instruction:\n{input}\n\n### Response:\n"


def encode_prompt_response(tokenizer, prompt, response, prompt_format="chat",
                           max_len=1024):
    """Tokenize prompt and response separately (avoids BPE seam ambiguity,
    matching train.py), concatenate, and return (ids, prompt_len)."""
    if prompt_format == "alpaca":
        prompt_ids = tokenizer(
            ALPACA_TEMPLATE.format(input=prompt), add_special_tokens=True).input_ids
    else:
        enc = build_chat_prompt(tokenizer, prompt, tokenize=True)
        prompt_ids = enc.input_ids if hasattr(enc, "input_ids") else enc
    response_ids = tokenizer(response, add_special_tokens=False).input_ids
    full_ids = (prompt_ids + response_ids)[:max_len]
    prompt_len = min(len(prompt_ids), len(full_ids))
    if prompt_len >= len(full_ids):
        raise ValueError(
            f"prompt fills max_len={max_len}, no response positions left "
            f"(prompt_len={prompt_len})")
    return full_ids, prompt_len


@torch.no_grad()
def extract_layer_activations(llm, tokenizer, layer, prompt, response,
                              prompt_format="chat", max_len=1024):
    """Teacher-force (prompt, response) and capture layer-`layer` output.

    Returns:
        h:             [1, S, d] float32 activations (full sequence)
        response_mask: [1, S] float, 1 on response positions
        prompt_len:    int
    """
    device = next(llm.parameters()).device
    full_ids, prompt_len = encode_prompt_response(
        tokenizer, prompt, response, prompt_format, max_len)
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)

    captured = {}

    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        captured["h"] = h.detach().float()

    handle = get_text_decoder(llm).layers[layer].register_forward_hook(hook)
    try:
        llm(input_ids=input_ids)
    finally:
        handle.remove()

    h = captured["h"]
    response_mask = torch.zeros(1, h.size(1), device=device)
    response_mask[0, prompt_len:] = 1.0
    return h, response_mask, prompt_len


def masked_mean(h, mask):
    """Mean-pool [B, S, d] over positions where mask [B, S] is 1 -> [B, d]."""
    m = mask.to(h.dtype).unsqueeze(-1)
    return (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


def _rbf_mmd2(x, y, bandwidths=None):
    """Biased MMD^2 between point clouds x [n, d], y [m, d] with a
    multi-scale RBF kernel (median-heuristic base bandwidth)."""
    z = torch.cat([x, y], dim=0)
    d2 = torch.cdist(z, z).pow(2)
    with torch.no_grad():
        med = d2[d2 > 0].median() if (d2 > 0).any() else torch.tensor(1.0, device=z.device)
    if bandwidths is None:
        bandwidths = (0.5, 1.0, 2.0)
    n = x.size(0)
    k = sum(torch.exp(-d2 / (b * med.clamp(min=1e-6))) for b in bandwidths)
    kxx, kyy, kxy = k[:n, :n], k[n:, n:], k[:n, n:]
    return kxx.mean() + kyy.mean() - 2.0 * kxy.mean()


def activation_distance(h_x, mask_x, h_y, mask_y, kind="mean_l2"):
    """Distance between two activation point clouds ([1, S, d] + [1, S] masks).

    mean_l2: squared L2 between masked mean-pooled vectors (per-dim mean).
    mmd:     multi-scale RBF MMD^2 between the masked point clouds.
    Differentiable in h_x (used as the inverse-solver objective).
    """
    if kind == "mean_l2":
        mx = masked_mean(h_x, mask_x)
        my = masked_mean(h_y, mask_y)
        return (mx - my).pow(2).mean()
    if kind == "mmd":
        x = h_x[0][mask_x[0] > 0]
        y = h_y[0][mask_y[0] > 0]
        return _rbf_mmd2(x, y)
    raise ValueError(f"unknown distance kind: {kind!r}")
