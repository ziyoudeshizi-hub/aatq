#!/usr/bin/env python
"""AATQ 7B experiment — cluster-ready, zero local dependencies.

Usage:
    # Quick test (1.1B, ~5 min)
    python scripts/run_cluster_7b.py --model tinyllama --steps 200 --lr 0.02

    # Full 7B experiment (~30 min)
    python scripts/run_cluster_7b.py --model llama2-7b --steps 500 --lr 0.01

    # With lm-eval (adds ~1h)
    python scripts/run_cluster_7b.py --model llama2-7b --steps 500 --lr 0.01 --eval

Models supported:
    tinyllama  → TinyLlama/TinyLlama-1.1B-Chat-v1.0  (sanity check)
    llama2-7b  → meta-llama/Llama-2-7b-hf
    llama3-8b  → meta-llama/Meta-Llama-3-8B
    qwen2-7b   → Qwen/Qwen2-7B
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone, timedelta

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── model registry ──────────────────────────────────────────────────────────
MODELS = {
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "llama2-7b": "meta-llama/Llama-2-7b-hf",
    "llama3-8b": "meta-llama/Meta-Llama-3-8B",
    "qwen2-7b":  "Qwen/Qwen2-7B",
}

# ── STQ_Group (inline, same as run_lm_eval.py) ──────────────────────────────
class STQ_Group(nn.Module):
    """Group-wise Soft Ternary Quantizer."""

    def __init__(self, linear: nn.Linear, group_size: int = 128):
        super().__init__()
        w = linear.weight.data.float()
        O, I = w.shape
        assert I % group_size == 0, f"in_features={I} not divisible by gs={group_size}"
        self.gs = group_size
        self.G = I // group_size
        self.O, self.I = O, I
        self.register_buffer("orig_w", w.half())
        w_3d = w.view(O, self.G, group_size)
        d = 0.7 * w_3d.std(dim=2)
        self.delta = nn.Parameter(d)
        d_3d = d.unsqueeze(-1)
        above = w_3d.abs() > d_3d
        a = (w_3d.abs() * above.float()).sum(dim=2) / above.float().sum(dim=2).clamp(min=1)
        self.alpha = nn.Parameter(a)
        self.tau = 1.0
        self.hard = False
        self.bias = nn.Parameter(linear.bias.data.clone()) if linear.bias is not None else None

    def forward(self, x):
        w_3d = self.orig_w.view(self.O, self.G, self.gs)
        delta_3d = self.delta.half().unsqueeze(-1)
        alpha_3d = self.alpha.half().unsqueeze(-1)
        abs_w = w_3d.abs()
        sgn_w = w_3d.sign()
        if self.hard:
            mask = (abs_w > delta_3d).half()
        else:
            mask_hard = (abs_w > delta_3d).half()
            sig_in = ((abs_w - delta_3d) / max(self.tau, 0.25)).clamp(-10, 10)
            mask_soft = torch.sigmoid(sig_in)
            mask = mask_hard.detach() + mask_soft - mask_soft.detach()
        wq = (alpha_3d * mask * sgn_w).view(self.O, self.I)
        return F.linear(x, wq, self.bias.half() if self.bias is not None else None)

    def get_hard_weight(self):
        w_3d = self.orig_w.view(self.O, self.G, self.gs)
        delta_3d = self.delta.detach().half().unsqueeze(-1)
        alpha_3d = self.alpha.detach().half().unsqueeze(-1)
        mask = (w_3d.abs() > delta_3d).half()
        return (alpha_3d * mask * w_3d.sign()).view(self.O, self.I)

    def sparsity(self):
        wq = self.get_hard_weight()
        return (wq == 0).float().mean().item()


# ── quantize ────────────────────────────────────────────────────────────────
def apply_quantization(model: nn.Module, group_size: int = 128) -> tuple[nn.Module, int]:
    """Replace all eligible nn.Linear with STQ_Group."""
    skip_names = {"lm_head", "embed_tokens", "rotary_emb"}
    count = 0
    param_count = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if module.weight.shape[0] < 64:
            continue
        if module.weight.shape[1] % group_size != 0:
            continue
        parts = name.split(".")
        if parts[-1] in skip_names:
            continue
        # navigate to parent
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        q = STQ_Group(module, group_size)
        setattr(parent, parts[-1], q)
        count += 1
        param_count += q.delta.numel() + q.alpha.numel()
    print(f"  Quantized {count} layers, {param_count:,} trainable params (gs={group_size})")
    return model, param_count


# ── data ────────────────────────────────────────────────────────────────────
def make_calibration_batches(tokenizer, n_samples: int = 128, seq_len: int = 256):
    """Use WikiText-2 train via HF datasets, fall back to random if offline."""
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        text = "".join([t for t in ds["text"] if t])
        tokens = tokenizer(text, return_tensors="pt").input_ids[0]
    except Exception:
        print("  datasets not available, using random calibration (suboptimal)")
        tokens = torch.randint(0, tokenizer.vocab_size or 32000, (n_samples * seq_len,))

    batches = []
    for i in range(n_samples):
        start = i * seq_len
        if start + seq_len > len(tokens):
            break
        batches.append(tokens[start:start + seq_len].unsqueeze(0).cuda())
    print(f"  {len(batches)} calibration batches (seq_len={seq_len})")
    return batches


# ── evaluate ────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_ppl(model: nn.Module, tokenizer, device: str,
                 num_sequences: int = 40, seq_len: int = 256) -> float:
    """PPL on WikiText-2 test set."""
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "".join([t for t in ds["text"] if t])
        tokens = tokenizer(text, return_tensors="pt").input_ids[0]
    except Exception:
        print("  Cannot load test data, returning -1")
        return -1.0

    model.eval()
    total_nll, total_tokens = 0.0, 0
    for i in range(num_sequences):
        start = i * seq_len
        if start + seq_len > len(tokens):
            break
        ids = tokens[start:start + seq_len].unsqueeze(0).to(device)
        out = model(ids, labels=ids)
        total_nll += out.loss.item() * ids.numel()
        total_tokens += ids.numel()
    return float(torch.exp(torch.tensor(total_nll / total_tokens)).item())


# ── train ───────────────────────────────────────────────────────────────────
def train_ste(model: nn.Module, batches: list, steps: int = 500, lr: float = 0.01):
    """STE group-wise training."""
    params = []
    for m in model.modules():
        if isinstance(m, STQ_Group):
            params.extend([m.delta, m.alpha])

    opt = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.99), eps=1e-4)
    model.gradient_checkpointing_enable()
    torch.cuda.empty_cache()

    best_loss, best_state = float("inf"), None
    t0 = time.time()

    for step in range(steps):
        # cosine tau 1.0 → 0.25
        progress = step / max(steps - 1, 1)
        tau = 0.25 + 0.5 * 0.75 * (1 + math.cos(math.pi * progress))
        for m in model.modules():
            if isinstance(m, STQ_Group):
                m.tau = tau

        batch = batches[step % len(batches)]
        out = model(batch, labels=batch)

        if torch.isnan(out.loss) or torch.isinf(out.loss):
            print(f"  NaN/Inf at step {step}, restoring best (loss={best_loss:.3f})")
            if best_state:
                _restore_best(model, best_state)
            break

        opt.zero_grad()
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)

        if step < 20:
            factor = (step + 1) / 20
            for pg in opt.param_groups:
                pg["lr"] = lr * factor

        opt.step()

        # clamp
        with torch.no_grad():
            for m in model.modules():
                if isinstance(m, STQ_Group):
                    m.delta.data.clamp_(min=1e-6, max=5.0)
                    m.alpha.data.clamp_(min=1e-6, max=5.0)

        loss_val = out.loss.item()
        if loss_val < best_loss:
            best_loss = loss_val
            best_state = _save_best(model)

        if step % 100 == 0 or step == steps - 1:
            elapsed = time.time() - t0
            d_mean = torch.stack([m.delta.mean() for m in model.modules() if isinstance(m, STQ_Group)]).mean().item()
            a_mean = torch.stack([m.alpha.mean() for m in model.modules() if isinstance(m, STQ_Group)]).mean().item()
            print(f"  {step:4d}/{steps} tau={tau:.3f} loss={loss_val:.4f} Δ={d_mean:.5f} α={a_mean:.5f} [{elapsed:.0f}s]")

    # restore best
    model.gradient_checkpointing_disable()
    if best_state:
        _restore_best(model, best_state)
        print(f"  Restored best checkpoint (loss={best_loss:.4f})")
    return model, best_loss


def _save_best(model):
    state = {}
    for m in model.modules():
        if isinstance(m, STQ_Group):
            state[id(m)] = {
                "delta": m.delta.data.detach().cpu().clone(),
                "alpha": m.alpha.data.detach().cpu().clone(),
            }
    return state


def _restore_best(model, state):
    for m in model.modules():
        if isinstance(m, STQ_Group):
            k = id(m)
            if k in state:
                m.delta.data.copy_(state[k]["delta"])
                m.alpha.data.copy_(state[k]["alpha"])


# ── main ────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="AATQ 7B cluster experiment")
    p.add_argument("--model", default="llama2-7b", choices=list(MODELS))
    p.add_argument("--gs", type=int, default=128)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--cal-samples", type=int, default=128)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--eval", action="store_true", help="Run lm-eval after training")
    p.add_argument("--eval-tasks", default="arc_easy,arc_challenge,boolq,hellaswag,piqa,winogrande")
    p.add_argument("--output-dir", default="./cluster_results")
    p.add_argument("--tag", default=None, help="Experiment tag (default: auto-generated)")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda"
    model_id = MODELS[args.model]
    tag = args.tag or f"{args.model}_gs{args.gs}_lr{args.lr}_s{args.steps}"
    os.makedirs(args.output_dir, exist_ok=True)

    tz = timezone(timedelta(hours=8))
    start_time = datetime.now(tz).isoformat()

    print("=" * 60)
    print(f"  AATQ 7B Experiment — {tag}")
    print(f"  Model:   {model_id}")
    print(f"  Config:  gs={args.gs}, steps={args.steps}, lr={args.lr}")
    print(f"  Start:   {start_time}")
    print(f"  GPU:     {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_mem/1e9:.1f}GB)")
    print("=" * 60)

    # ── 1. Load model ──────────────────────────────────────────────────────
    print("\n[1/4] Loading model …")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True,
    )
    model.eval()
    print(f"  GPU memory: {torch.cuda.memory_allocated()/1e9:.2f}GB allocated")

    # ── 2. FP16 baseline ───────────────────────────────────────────────────
    print("\n[2/4] FP16 baseline PPL …")
    fp16_ppl = evaluate_ppl(model, tokenizer, device)
    print(f"  FP16 PPL = {fp16_ppl:.2f}")

    # ── 3. Quantize + Train ────────────────────────────────────────────────
    print("\n[3/4] Applying group-wise quantization + STE training …")
    model, n_params = apply_quantization(model, group_size=args.gs)
    batches = make_calibration_batches(tokenizer, n_samples=args.cal_samples, seq_len=args.seq_len)

    t_train = time.time()
    model, best_loss = train_ste(model, batches, steps=args.steps, lr=args.lr)
    train_time = time.time() - t_train
    print(f"  Training done in {train_time:.0f}s")

    # ── 4. Evaluate ────────────────────────────────────────────────────────
    print("\n[4/4] Evaluating hard ternary …")
    for m in model.modules():
        if isinstance(m, STQ_Group):
            m.hard = True

    ternary_ppl = evaluate_ppl(model, tokenizer, device)
    sp = sum(m.sparsity() for m in model.modules() if isinstance(m, STQ_Group))
    nq = sum(1 for m in model.modules() if isinstance(m, STQ_Group))
    sparsity = sp / max(nq, 1)

    # delta/alpha stats
    deltas = torch.stack([m.delta.mean() for m in model.modules() if isinstance(m, STQ_Group)])
    alphas = torch.stack([m.alpha.mean() for m in model.modules() if isinstance(m, STQ_Group)])

    # ── Report ─────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  Model:            {model_id}")
    print(f"  FP16 PPL:         {fp16_ppl:.2f}")
    print(f"  Ternary PPL:      {ternary_ppl:.2f}")
    print(f"  Improvement:      {(1 - ternary_ppl/fp16_ppl)*100 if fp16_ppl > 0 else 0:.1f}% of FP16")
    print(f"  Sparsity:         {sparsity:.1%}")
    print(f"  δ (mean±std):     {deltas.mean().item():.5f} ± {deltas.std().item():.5f}")
    print(f"  α (mean±std):     {alphas.mean().item():.5f} ± {alphas.std().item():.5f}")
    print(f"  Train time:       {train_time:.0f}s")
    print(f"  Train params:     {n_params:,}")
    print(f"  Go/No-Go (PPL<50): {'PASS' if ternary_ppl < 50 else 'NOT YET'}")

    # ── Save JSON ──────────────────────────────────────────────────────────
    result = {
        "tag": tag,
        "model": model_id,
        "group_size": args.gs,
        "steps": args.steps,
        "lr": args.lr,
        "cal_samples": args.cal_samples,
        "seq_len": args.seq_len,
        "fp16_ppl": round(fp16_ppl, 2),
        "ternary_ppl": round(ternary_ppl, 2),
        "sparsity": round(sparsity, 4),
        "delta_mean": round(deltas.mean().item(), 6),
        "delta_std": round(deltas.std().item(), 6),
        "alpha_mean": round(alphas.mean().item(), 6),
        "alpha_std": round(alphas.std().item(), 6),
        "train_time_s": round(train_time, 1),
        "trainable_params": n_params,
        "gpu": torch.cuda.get_device_name(0),
        "start_time": start_time,
        "go": ternary_ppl < 50,
    }

    out_path = os.path.join(args.output_dir, f"{tag}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n  Results → {out_path}")

    # ── Optional lm-eval ───────────────────────────────────────────────────
    if args.eval:
        print("\n  Running lm-eval (this takes a while) …")
        try:
            import lm_eval
            results = lm_eval.simple_evaluate(
                model="hf",
                model_args=f"pretrained={model_id},dtype=float16",
                tasks=args.eval_tasks.split(","),
                num_fewshot=0,
                batch_size="auto",
            )
            eval_summary = {}
            for task, metrics in results["results"].items():
                acc = metrics.get("acc,none", metrics.get("acc_norm,none", None))
                if acc is not None:
                    eval_summary[task] = round(float(acc) * 100, 1)
                    print(f"    {task}: {eval_summary[task]:.1f}%")
            result["lm_eval"] = eval_summary
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"  lm-eval failed: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
