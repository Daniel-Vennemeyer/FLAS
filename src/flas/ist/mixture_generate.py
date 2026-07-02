"""Multi-concept steered generation (causal validation of inverse solutions).

MixtureFlasGenerator extends FlasGenerator with a joint-Euler mixture hook:

    h_{k+1} = h_k + sum_i (alpha_i / N) * v_theta(h_k, k*alpha_i/N, c_i)

matching TransportMixture, so strengths inferred by the inverse solver can be
applied verbatim during generation. Self-attn KV caches are kept per
(euler_step, concept): within a step, every concept's velocity is computed
from the same h_k, but each concept's FlowBlock sees its own cross-attended
stream, so caches cannot be shared across concepts.
"""

import torch

from flas.generate import FlasGenerator, load_generator
from flas.model import build_chat_prompt


class MixtureFlasGenerator(FlasGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mix_hiddens = None   # list of [1, L_i, d]
        self._mix_masks = None     # list of [1, L_i]
        self._mix_alphas = None    # list of float
        self._mix_caches = None    # _mix_caches[step][concept] = per-block caches

    def _hook_fn(self, module, input, output):
        if self._mix_alphas is None:
            return super()._hook_fn(module, input, output)  # single-concept path
        if not self._active:
            return output
        is_tuple = isinstance(output, tuple)
        h_orig = output[0] if is_tuple else output
        h = h_orig.to(self._flow_dtype)
        bsz = h.size(0)
        n = self._n_steps

        for k in range(n):
            v_total = torch.zeros_like(h)
            for i, alpha in enumerate(self._mix_alphas):
                if alpha == 0:
                    continue
                dt = alpha / n
                t_k = torch.full((bsz,), dt * k, device=h.device)
                kwargs = dict(
                    t=t_k, padding_mask=self._padding_mask[:bsz],
                    use_cache=True, position_ids=self._position_ids[:bsz])
                if self._is_prefill:
                    v, caches = self.flow_fn(
                        h, self._mix_hiddens[i].expand(bsz, -1, -1),
                        self._mix_masks[i].expand(bsz, -1),
                        past_len=0, **kwargs)
                else:
                    v, caches = self.flow_fn(
                        h, self._mix_hiddens[i].expand(bsz, -1, -1),
                        self._mix_masks[i].expand(bsz, -1),
                        self_attn_caches=self._mix_caches[k][i],
                        past_len=self._past_len, **kwargs)
                self._mix_caches[k][i] = caches
                v_total = v_total + dt * v
            h = h + v_total

        h_out = h.to(h_orig.dtype)
        return (h_out,) + output[1:] if is_tuple else h_out

    @torch.no_grad()
    def generate_mixture(self, prompts, concept_texts, alphas, n_steps=3,
                         max_tokens=128, temperature=1.0, max_batch=16,
                         enable_thinking=False):
        """Generate with a fixed mixture of concepts applied to every prompt.

        Args:
            prompts:       list of user instructions
            concept_texts: list of M concept descriptions
            alphas:        list of M strengths (flow times); 0 disables a concept
        Returns a list of dicts {prompt, prompt_idx, generation}.
        """
        assert len(concept_texts) == len(alphas)
        self._n_steps = n_steps
        self._mix_alphas = [float(a) for a in alphas]
        self._mix_hiddens, self._mix_masks = [], []
        for c in concept_texts:
            hid, mask = self.encode_concept(c)
            self._mix_hiddens.append(hid)
            self._mix_masks.append(mask)

        results = [None] * len(prompts)
        try:
            for chunk_start in range(0, len(prompts), max_batch):
                chunk = prompts[chunk_start:chunk_start + max_batch]
                bsz = len(chunk)
                if getattr(self, "_prompt_format", "chat") == "alpaca":
                    formatted = []
                    for p in chunk:
                        fmt = f"### Instruction:\n{p}\n\n### Response:\n"
                        if self.tokenizer.bos_token:
                            fmt = self.tokenizer.bos_token + fmt
                        formatted.append(fmt)
                else:
                    formatted = [
                        build_chat_prompt(self.tokenizer, p, tokenize=False,
                                          enable_thinking=enable_thinking)
                        for p in chunk]

                enc = self.tokenizer(
                    formatted, return_tensors="pt", padding=True,
                    truncation=True, max_length=512,
                    add_special_tokens=False).to("cuda")
                input_ids = enc.input_ids
                attention_mask = enc.attention_mask
                prompt_len = input_ids.shape[1]

                self._padding_mask = attention_mask.float()
                self._mix_caches = [
                    [None] * len(self._mix_alphas) for _ in range(n_steps)]
                self._is_prefill = True
                self._past_len = 0
                position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)
                self._position_ids = position_ids

                self._install_hook()
                self._active = True

                out = self.llm(input_ids, attention_mask=attention_mask,
                               position_ids=position_ids, use_cache=True)
                past_kv = out.past_key_values
                next_logits = out.logits[:, -1, :]

                self._is_prefill = False
                self._past_len = prompt_len

                generated = input_ids
                unfinished = torch.ones(bsz, dtype=torch.bool, device="cuda")
                for _ in range(max_tokens):
                    if temperature > 0:
                        probs = torch.softmax(next_logits / temperature, dim=-1)
                        next_token = torch.multinomial(probs, 1)
                    else:
                        next_token = next_logits.argmax(dim=-1, keepdim=True)
                    next_token = next_token.masked_fill(
                        ~unfinished.unsqueeze(1), self.tokenizer.pad_token_id)
                    generated = torch.cat([generated, next_token], dim=1)
                    attention_mask = torch.cat(
                        [attention_mask, unfinished.unsqueeze(1).long()], dim=1)
                    eos_hit = (next_token.squeeze(1) == self.tokenizer.eos_token_id)
                    unfinished = unfinished & ~eos_hit
                    if not unfinished.any():
                        break

                    self._padding_mask = attention_mask.float()
                    position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)
                    self._position_ids = position_ids[:, -1:]
                    out = self.llm(next_token, attention_mask=attention_mask,
                                   position_ids=self._position_ids,
                                   past_key_values=past_kv, use_cache=True)
                    past_kv = out.past_key_values
                    next_logits = out.logits[:, -1, :]
                    self._past_len += 1

                self._active = False
                for i in range(bsz):
                    gen_ids = generated[i, prompt_len:]
                    text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                    results[chunk_start + i] = {
                        "prompt": chunk[i],
                        "prompt_idx": chunk_start + i,
                        "generation": text,
                    }
                del past_kv, out
                self._mix_caches = None
                torch.cuda.empty_cache()
        finally:
            self._active = False
            self._mix_alphas = None
            self._mix_caches = None
            self._remove_hook()
        return results


def load_mixture_generator(flow_ckpt, model_id=None, layer=None, num_blocks=None):
    """Load a MixtureFlasGenerator by rebinding a load_generator() result."""
    gen = load_generator(flow_ckpt, model_id=model_id, layer=layer,
                         num_blocks=num_blocks)
    mix = MixtureFlasGenerator(gen.llm, gen.tokenizer, gen.flow_fn,
                               gen.concept_enc, gen.layer)
    mix._prompt_format = getattr(gen, "_prompt_format", "chat")
    return mix
