"""Graded strength<->flow-time training for FLAS.

Instead of sampling one flow time T ~ Uniform per batch (flas.train), each
sample carries an ordinal concept strength s in {1, 2, 3} (low/med/high) and
is supervised at T = t_unit * s (with multiplicative jitter), teacher-forcing
the response written at that strength. This calibrates flow time as concept
intensity with behavioral (LM-loss) supervision only — no activation-matching
target, so the ill-posed token alignment between different-strength responses
never arises.

Data: parquet with columns input / output / output_concept / concept_id /
strength (int >= 0). strength-0 (neutral) rows are dropped: at T = 0 the flow
is the identity and contributes no gradient. Build the parquet with
scripts/ist_build_graded_data.py.

Usage:
    python -m flas.ist.train_graded \
        --data-dir data/graded --output-dir checkpoints --run-name ist_graded \
        --t-unit 1.0 --t-jitter 0.25 --n-steps 3
"""

import argparse
import json
from pathlib import Path
from functools import partial

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import AutoTokenizer

from flas.model import get_text_decoder, build_chat_prompt
from flas.train import FlasModule, ALPACA_TEMPLATE


class GradedSteeringDataset(Dataset):
    def __init__(self, df):
        self.samples = [
            (row["input"], row["output"], row["output_concept"],
             row["concept_id"], float(row["strength"]))
            for _, row in df.iterrows()
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def graded_collate(batch, tokenizer, max_len, concept_max_len,
                   prompt_format="chat"):
    """flas.train.collate_fn with a strength column threaded through (the
    base collate may skip degenerate samples, so strengths must be filtered
    in lockstep — hence the copy rather than a wrapper)."""
    input_texts, output_texts, concept_texts, cids, strengths = zip(*batch)

    full_ids_list, prompt_lens = [], []
    keep_concepts, keep_cids, keep_strengths = [], [], []
    for inp, out, ct, cid, s in zip(input_texts, output_texts, concept_texts,
                                    cids, strengths):
        if not ct or not ct.strip():
            continue
        if prompt_format == "alpaca":
            prompt_ids = tokenizer(
                ALPACA_TEMPLATE.format(input=inp), add_special_tokens=True).input_ids
        else:
            prompt_enc = build_chat_prompt(tokenizer, inp, tokenize=True)
            prompt_ids = prompt_enc.input_ids if hasattr(prompt_enc, "input_ids") else prompt_enc
        output_ids = tokenizer(out, add_special_tokens=False).input_ids
        full_ids = (prompt_ids + output_ids)[:max_len]
        prompt_len = min(len(prompt_ids), len(full_ids))
        if prompt_len <= 0 or prompt_len >= len(full_ids):
            continue
        full_ids_list.append(full_ids)
        prompt_lens.append(prompt_len)
        keep_concepts.append(ct)
        keep_cids.append(cid)
        keep_strengths.append(s)

    if not full_ids_list:
        raise RuntimeError("graded_collate: entire batch was skipped — "
                           "increase --max-len")

    enc = tokenizer.pad({"input_ids": full_ids_list}, return_tensors="pt",
                        padding=True)
    input_ids = enc.input_ids
    attention_mask = enc.attention_mask

    labels = input_ids.clone()
    for i, plen in enumerate(prompt_lens):
        labels[i, :plen] = -100
    labels[attention_mask == 0] = -100

    concept_enc = tokenizer(keep_concepts, return_tensors="pt", padding=True,
                            truncation=True, max_length=concept_max_len)

    return (input_ids, attention_mask, labels,
            concept_enc.input_ids, concept_enc.attention_mask,
            torch.tensor(keep_cids, dtype=torch.long),
            torch.tensor(keep_strengths, dtype=torch.float32))


class GradedFlasModule(FlasModule):
    """FlasModule with per-sample flow time T_i = t_unit * strength_i."""

    def forward_with_flow(self, input_ids, attention_mask, labels,
                          concept_input_ids, concept_attention_mask, T_total):
        """Identical to FlasModule.forward_with_flow except T_total is a
        per-sample [bsz] tensor rather than a scalar."""
        concept_hidden = self.concept_enc(concept_input_ids, concept_attention_mask)
        velocity_capture = {}
        n_steps = self.args.n_steps

        concept_mask_f = concept_attention_mask.float()
        attn_mask_f = attention_mask.float()

        def hook(module, input, output):
            is_tuple = isinstance(output, tuple)
            h_orig = output[0] if is_tuple else output
            h = h_orig.float()
            bsz = h.size(0)
            dt = T_total[:bsz].to(h.device) / n_steps  # [bsz]
            last_v = None
            for k in range(n_steps):
                t_k = dt * k
                v, _ = self.flow_fn(
                    h, concept_hidden, concept_mask_f,
                    t=t_k, padding_mask=attn_mask_f)
                h = h + dt[:, None, None] * v
                last_v = v
            velocity_capture["v"] = last_v
            m = attn_mask_f.unsqueeze(-1)
            n_real = m.sum(dim=1).clamp(min=1)
            v_per = (last_v.detach().norm(dim=-1, keepdim=True) * m).sum(dim=1) / n_real
            h_per = (h.detach().norm(dim=-1, keepdim=True) * m).sum(dim=1) / n_real
            velocity_capture["v_norm"] = v_per.mean()
            velocity_capture["h_norm"] = h_per.mean()
            h_out = h.to(h_orig.dtype)
            return (h_out,) + output[1:] if is_tuple else h_out

        handle = get_text_decoder(self.llm).layers[self.args.layer].register_forward_hook(hook)
        outputs = self.llm(input_ids=input_ids, attention_mask=attention_mask,
                           labels=labels)
        handle.remove()
        return outputs.loss, velocity_capture

    def _sample_graded_T(self, strengths):
        j = self.args.t_jitter
        u = 1.0 + (torch.rand_like(strengths) * 2.0 - 1.0) * j
        return self.args.t_unit * strengths * u

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        sch = self.lr_schedulers()

        input_ids, attn_mask, labels, c_ids, c_mask, cids, strengths = batch
        T_total = self._sample_graded_T(strengths)

        lm_loss, velocity_capture = self.forward_with_flow(
            input_ids, attn_mask, labels, c_ids, c_mask, T_total)
        velocity = velocity_capture.get("v")
        div_loss = self.compute_diversity_loss(velocity, cids, attn_mask)
        loss = (lm_loss + self.args.div_weight * div_loss) / self.args.grad_accum

        self.manual_backward(loss)

        if (batch_idx + 1) % self.args.grad_accum == 0:
            trainable = [p for p in self.flow_fn.parameters() if p.requires_grad]
            trainable += [p for p in self.concept_enc.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            sch.step()
            opt.zero_grad()

        self.log("train/lm_loss", lm_loss, prog_bar=True)
        self.log("train/div_loss", div_loss)
        self.log("train/loss", loss)
        self.log("train/mean_T", T_total.mean())
        self.log("train/lr", sch.get_last_lr()[0])
        if "v_norm" in velocity_capture:
            self.log("train/v_norm", velocity_capture["v_norm"])
            self.log("train/vh_ratio",
                     velocity_capture["v_norm"] / velocity_capture["h_norm"])

    def validation_step(self, batch, batch_idx):
        input_ids, attn_mask, labels, c_ids, c_mask, cids, strengths = batch
        T_total = self.args.t_unit * strengths  # no jitter at validation
        lm_loss, _ = self.forward_with_flow(
            input_ids, attn_mask, labels, c_ids, c_mask, T_total)
        self.log("val/loss", lm_loss, prog_bar=True, sync_dist=True)
        self.log("val_loss", lm_loss)
        return lm_loss


def prepare_graded_data(args, tokenizer):
    for cand in ("train_data.parquet", "train/data.parquet", "train.parquet"):
        train_parquet = Path(args.data_dir) / cand
        if train_parquet.exists():
            break
    df = pd.read_parquet(train_parquet)
    if "output_concept" not in df.columns and "concept" in df.columns:
        df = df.rename(columns={"concept": "output_concept"})
    if "strength" not in df.columns:
        raise ValueError(
            f"{train_parquet} has no 'strength' column — build graded data "
            "with scripts/ist_build_graded_data.py")

    n_neutral = int((df["strength"] <= 0).sum())
    df = df[df["strength"] > 0].reset_index(drop=True)
    print(f"Dropped {n_neutral} strength-0 (neutral) rows: T=0 is the "
          f"identity flow and gives no gradient")

    all_cids = sorted(df["concept_id"].unique())
    if getattr(args, "heldout_ids_file", None):
        ho_cids = set(json.load(open(args.heldout_ids_file)))
        df = df[~df["concept_id"].isin(ho_cids)].reset_index(drop=True)
        print(f"Held out {len(ho_cids)} concepts from {args.heldout_ids_file}")
    elif args.val_n_concepts > 0:
        n_ho = min(args.val_n_concepts, len(all_cids) - 1)
        torch.manual_seed(42)
        perm = torch.randperm(len(all_cids)).tolist()
        ho_cids = set(all_cids[i] for i in perm[:n_ho])
        df = df[~df["concept_id"].isin(ho_cids)].reset_index(drop=True)
        print(f"Held out {n_ho} concepts for evaluation")

    n_val = min(args.n_val_samples, len(df) // 10)
    val_idx = torch.randperm(len(df))[:n_val].tolist()
    val_mask = torch.zeros(len(df), dtype=torch.bool)
    val_mask[val_idx] = True
    val_df = df.iloc[val_idx].reset_index(drop=True)
    train_df = df.iloc[~val_mask.numpy()].reset_index(drop=True)

    print(f"Train: {len(train_df)} samples ({train_df.concept_id.nunique()} "
          f"concepts, strengths {sorted(train_df.strength.unique())})")
    print(f"Val:   {len(val_df)} samples")

    collate = partial(graded_collate, tokenizer=tokenizer, max_len=args.max_len,
                      concept_max_len=args.concept_max_len,
                      prompt_format=args.prompt_format)
    train_loader = DataLoader(
        GradedSteeringDataset(train_df), batch_size=args.batch_size,
        shuffle=True, drop_last=True, collate_fn=collate,
        num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
        pin_memory=True)
    val_loader = DataLoader(
        GradedSteeringDataset(val_df), batch_size=args.batch_size,
        shuffle=True, collate_fn=collate,
        num_workers=min(2, args.num_workers),
        persistent_workers=args.num_workers > 0,
        pin_memory=True)
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--model-id", type=str, default="google/gemma-2-2b-it")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--run-name", type=str, default="ist_graded")
    parser.add_argument("--num-blocks", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--enc-lr", type=float, default=1e-5)
    parser.add_argument("--div-weight", type=float, default=0.1)
    parser.add_argument("--total-steps", type=int, default=80000)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--val-every", type=int, default=500)
    parser.add_argument("--val-n-concepts", type=int, default=0)
    parser.add_argument("--heldout-ids-file", type=str, default=None)
    parser.add_argument("--n-val-samples", type=int, default=100)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--concept-max-len", type=int, default=64)
    # Graded supervision: T_i = t_unit * strength_i * U[1-j, 1+j]
    parser.add_argument("--t-unit", type=float, default=1.0,
                        help="Flow time per strength level (strength s trains at T ~= s * t_unit)")
    parser.add_argument("--t-jitter", type=float, default=0.25,
                        help="Multiplicative jitter half-width on the strength-mapped T")
    parser.add_argument("--n-steps", type=int, default=3)
    parser.add_argument("--no-gemma-mlp-init", action="store_true")
    parser.add_argument("--unfreeze-concept-enc", action="store_true")
    parser.add_argument("--disable-cross-attn", action="store_true")
    parser.add_argument("--disable-self-attn", action="store_true")
    parser.add_argument("--disable-mlp", action="store_true")
    parser.add_argument("--prompt-format", choices=["chat", "alpaca"], default="chat")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--val-batches", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_loader, val_loader = prepare_graded_data(args, tokenizer)
    model = GradedFlasModule(args)

    ckpt_dir = Path(args.output_dir) / args.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(ckpt_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    tb_logger = TensorBoardLogger(save_dir=str(ckpt_dir), name="tb_logs", version="")
    early_stop = EarlyStopping(monitor="val_loss", patience=args.patience, mode="min")

    trainer = pl.Trainer(
        max_steps=args.total_steps,
        val_check_interval=args.val_every,
        limit_val_batches=args.val_batches,
        callbacks=[early_stop],
        logger=tb_logger,
        default_root_dir=str(ckpt_dir),
        enable_progress_bar=True,
        gradient_clip_val=None,
        enable_checkpointing=False,
        accelerator="gpu",
        devices=1,
        precision=args.precision,
        log_every_n_steps=50,
    )
    trainer.fit(model, train_loader, val_loader)

    flow_sd = {k: v.to(torch.bfloat16) for k, v in model.flow_fn.state_dict().items()}
    final_ckpt = {"flow_fn": flow_sd}
    if args.unfreeze_concept_enc:
        final_ckpt["concept_enc"] = {
            k: v.to(torch.bfloat16) for k, v in model.concept_enc.state_dict().items()}
    torch.save(final_ckpt, ckpt_dir / "final.pt")
    print(f"Done. Saved to {ckpt_dir}", flush=True)


if __name__ == "__main__":
    main()
