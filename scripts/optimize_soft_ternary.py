#!/usr/bin/env python
"""Phase 1: Soft Ternary Optimization with Temperature Annealing.

Core experiment: can we optimize Δ and α via gradient descent on a smooth
sigmoid landscape, then anneal to hard ternary, WITHOUT overfitting to
the calibration set?

Uses Adam with eps=1e-4 (NOT default 1e-8) to prevent NaN from tiny gradients.
No gradient clipping — Adam's adaptive lr handles magnitude naturally.
Fallback: SGD with large lr (0.1) and no grad clip.

Usage:
    python scripts/optimize_soft_ternary.py
    python scripts/optimize_soft_ternary.py --optimizer sgd --lr 0.1
    python scripts/optimize_soft_ternary.py --kl-lambda 0.1
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from aatq.calibration import calibration_iterator, load_wikitext_samples
from aatq.eval import evaluate_perplexity_fast
from aatq.quantize import (
    apply_naive_ternary,
    collected_quantizer_params,
    replace_with_soft_ternary,
    set_all_hard_mode,
    set_all_tau,
)
from aatq.soft_ternary import SoftTernaryQuantizer, temperature_schedule
from aatq.utils import count_quantizable_layers, model_size_mb


# ═══════════════════════════════════════════════════════════════
# Argument parsing
# ═══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="AATQ Phase 1: Soft Ternary Optimization")
    p.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--steps", type=int, default=200,
                   help="Total optimization steps")
    p.add_argument("--lr", type=float, default=0.01,
                   help="Learning rate (Adam, aggressive for delta/alpha)")
    p.add_argument("--momentum", type=float, default=0.9,
                   help="SGD momentum (only used if --optimizer sgd)")
    p.add_argument("--optimizer", type=str, default="adam",
                   choices=["adam", "sgd"],
                   help="Optimizer choice: adam (recommended) or sgd")
    p.add_argument("--tau-start", type=float, default=1.0)
    p.add_argument("--tau-end", type=float, default=0.01)
    p.add_argument("--cal-samples", type=int, default=64,
                   help="Total calibration samples (split 80/20 train/val)")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--eval-every", type=int, default=20,
                   help="Evaluate val loss every N steps")
    p.add_argument("--ppl-every", type=int, default=100,
                   help="Evaluate hard ternary PPL every N steps")
    p.add_argument("--val-samples-for-ppl", type=int, default=8,
                   help="Number of val sequences for intermediate PPL checks")
    p.add_argument("--kl-lambda", type=float, default=0.0,
                   help="KL distillation weight (0 = no distillation)")
    p.add_argument("--kl-temperature", type=float, default=2.0)
    p.add_argument("--no-distill", action="store_true",
                   help="Force no distillation")
    p.add_argument("--skip-lm-head", action="store_true", default=True)
    p.add_argument("--skip-embed", action="store_true", default=True)
    p.add_argument("--skip-baseline", action="store_true", default=False,
                   help="Skip FP16 baseline eval (saves memory, use known value)")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════
# KL Distillation (optional)
# ═══════════════════════════════════════════════════════════════

def compute_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    student_log_prob = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_prob = F.softmax(teacher_logits / temperature, dim=-1)
    kl = F.kl_div(student_log_prob, teacher_prob, reduction="batchmean", log_target=False)
    return kl * (temperature ** 2)


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def train_val_split(samples: list[dict], train_ratio: float = 0.8):
    n_train = int(len(samples) * train_ratio)
    return samples[:n_train], samples[n_train:]


@torch.no_grad()
def compute_val_loss(
    model: nn.Module,
    val_samples: list[dict],
    device: str,
    max_batches: int = 8,
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in calibration_iterator(val_samples, batch_size=1, device=device):
        outputs = model(**batch, labels=batch["input_ids"])
        total_loss += outputs.loss.item()
        n_batches += 1
        if n_batches >= max_batches:
            break

    model.train()
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def compute_hard_ternary_ppl(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    device: str,
    num_sequences: int = 8,
    seq_len: int = 1024,
) -> float:
    was_training = model.training
    set_all_hard_mode(model, True)
    ppl = evaluate_perplexity_safe(model, tokenizer, num_sequences=num_sequences, seq_len=seq_len, device=device)
    set_all_hard_mode(model, False)
    if was_training:
        model.train()
    return ppl


@torch.no_grad()
def evaluate_perplexity_safe(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    num_sequences: int = 40,
    seq_len: int = 1024,
    device: str = "cuda",
) -> float:
    from aatq.calibration import load_wikitext_samples

    samples = load_wikitext_samples(
        tokenizer, split="test", num_samples=num_sequences, seq_len=seq_len, seed=0
    )

    model.eval()
    model = model.to(device)

    total_nll = 0.0
    total_tokens = 0

    for i, sample in enumerate(samples):
        input_ids = sample["input_ids"].unsqueeze(0).to(device)
        outputs = model(input_ids, labels=input_ids)
        total_nll += outputs.loss.item() * input_ids.numel()
        total_tokens += input_ids.numel()
        del outputs, input_ids
        if i < len(samples) - 1:
            torch.cuda.empty_cache()

    ppl = float(torch.exp(torch.tensor(total_nll / total_tokens)).item())
    return ppl


def avg_param(param: nn.Parameter) -> float:
    return param.data.abs().mean().item()


def count_quantizers(model: nn.Module) -> int:
    return sum(1 for m in model.modules() if isinstance(m, SoftTernaryQuantizer))


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    if args.no_distill:
        args.kl_lambda = 0.0

    use_distill = args.kl_lambda > 0.0

    print("=" * 60)
    print("  AATQ Phase 1 — Soft Ternary Optimization")
    print(f"  Model:       {args.model}")
    print(f"  Steps:       {args.steps}")
    print(f"  Optimizer:   {args.optimizer.upper()}, LR={args.lr}")
    print(f"  τ schedule:  {args.tau_start} → {args.tau_end} (cosine)")
    print(f"  Cal samples: {args.cal_samples} (80/20 train/val)")
    print(f"  Distillation: {'KL λ=' + str(args.kl_lambda) if use_distill else 'OFF (pure CE)'}")
    print(f"  Device:      {args.device}")
    print("=" * 60)

    # ── 1. Load model ──────────────────────────────────────────
    print("\n[1/6] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    num_layers = count_quantizable_layers(model)
    fp16_mb = model_size_mb(model)
    print(f"  Quantizable Linear layers: {num_layers}")
    print(f"  FP16 model size: {fp16_mb:.1f} MB")

    skip_names = set()
    if args.skip_lm_head:
        skip_names.add("lm_head")
    if args.skip_embed:
        skip_names.add("embed_tokens")

    # ── 2. FP16 baseline PPL ───────────────────────────────────
    if args.skip_baseline:
        fp16_ppl = 13.67  # known value from Phase 0
        print(f"\n[2/6] FP16 baseline (skipped, using known value): {fp16_ppl:.2f}")
    else:
        print("\n[2/6] FP16 baseline perplexity...")
        fp16_ppl = evaluate_perplexity_safe(
            model, tokenizer, num_sequences=20, seq_len=args.seq_len, device=args.device
        )
        print(f"  FP16 PPL: {fp16_ppl:.2f}")

    naive_ppl = 47771.0
    stat_init_ppl = 47771.0
    print(f"\n  Naive Ternary PPL (reference): {naive_ppl:.0f}")
    print(f"  Statistical Init PPL (reference): ~{stat_init_ppl:.0f}")

    # ── 3. Replace model with trainable quantizers ─────────────
    print("\n[3/6] Replacing Linear layers with SoftTernaryQuantizer...")
    model = replace_with_soft_ternary(model, skip_names=skip_names)
    set_all_tau(model, args.tau_start)
    set_all_hard_mode(model, False)

    n_quantizers = count_quantizers(model)
    quant_params = collected_quantizer_params(model)
    n_params = sum(p.numel() for p in quant_params)
    print(f"  Quantizers: {n_quantizers}")
    print(f"  Optimizable parameters (Δ + α): {n_params:,}")

    # ── 4. Load and split calibration data ─────────────────────
    print("\n[4/6] Loading calibration data (WikiText-2 train)...")
    all_samples = load_wikitext_samples(
        tokenizer,
        split="train",
        num_samples=args.cal_samples,
        seq_len=args.seq_len,
        seed=42,
    )
    train_samples, val_samples = train_val_split(all_samples, train_ratio=0.8)
    print(f"  Train samples: {len(train_samples)}")
    print(f"  Val samples:   {len(val_samples)}")

    # ── 5. Teacher model (optional) ────────────────────────────
    teacher = None
    if use_distill:
        print(f"  Loading teacher model for KL distillation...")
        teacher = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True,
        )
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        print(f"  Teacher on CPU (will move per-batch)")

    # ── 6. Optimization loop ───────────────────────────────────
    print(f"\n[6/6] Optimization loop ({args.steps} steps)...")
    print(f"  {'Step':>5s} {'τ':>6s} {'train_loss':>10s} {'val_loss':>10s} {'gap':>8s} {'avg_Δ':>8s} {'avg_α':>8s} {'hard_ppl':>10s}")
    print(f"  {'-'*5} {'-'*6} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")

    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            quant_params, lr=args.lr,
            betas=(0.9, 0.99),  # faster adaptation than default (0.9, 0.999)
            eps=1e-4,           # critical: prevent denominator → 0 for tiny params
        )
    else:
        optimizer = torch.optim.SGD(quant_params, lr=args.lr, momentum=args.momentum)
    warmup_steps = max(1, int(args.steps * 0.1))
    base_lr = args.lr

    train_batches = list(calibration_iterator(
        train_samples, batch_size=args.batch_size, device=args.device
    ))

    train_losses: list[float] = []
    val_losses: list[float] = []
    hard_ppls: list[tuple[int, float]] = []
    consecutive_val_rises = 0
    best_val_loss = float("inf")
    overfitting_warned = False

    step = 0
    batches_per_epoch = len(train_batches)
    epochs_needed = math.ceil(args.steps / batches_per_epoch)

    for epoch in range(epochs_needed):
        if step >= args.steps:
            break

        for batch in train_batches:
            if step >= args.steps:
                break

            tau = temperature_schedule(step, args.steps, args.tau_start, args.tau_end)
            set_all_tau(model, tau)

            # LR warmup
            if step < warmup_steps:
                lr = base_lr * (step + 1) / warmup_steps
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

            optimizer.zero_grad()

            outputs = model(**batch, labels=batch["input_ids"])
            ce_loss = outputs.loss
            total_loss = ce_loss

            if torch.isnan(ce_loss) or torch.isinf(ce_loss):
                print(f"\n  ⚠ Step {step}: NaN/Inf loss, zeroing grad and skipping")
                optimizer.zero_grad()
                del outputs
                step += 1
                continue

            # Optional KL distillation
            kl_loss_val = 0.0
            if use_distill and teacher is not None:
                input_ids = batch["input_ids"]
                teacher_device = next(model.parameters()).device
                teacher_cpu = teacher.to(teacher_device)
                with torch.no_grad():
                    teacher_outputs = teacher_cpu(input_ids, labels=input_ids)
                teacher_logits = teacher_outputs.logits.detach()
                teacher.to("cpu")
                torch.cuda.empty_cache()
                kl_loss_val = compute_kl_loss(
                    outputs.logits, teacher_logits, args.kl_temperature
                )
                total_loss = ce_loss + args.kl_lambda * kl_loss_val

            total_loss.backward()

            # Check for NaN/Inf gradients before stepping
            has_nan_grad = any(
                p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
                for p in quant_params
            )
            if has_nan_grad:
                print(f"\n  ⚠ Step {step}: NaN/Inf gradient, skipping")
                optimizer.zero_grad()
                del outputs
                step += 1
                continue

            optimizer.step()

            del outputs

            # Clamp Δ and α to safe ranges
            with torch.no_grad():
                for m in model.modules():
                    if isinstance(m, SoftTernaryQuantizer):
                        m.delta.data.clamp_(min=1e-6, max=5.0)
                        m.alpha.data.clamp_(min=1e-6, max=5.0)

            train_losses.append(ce_loss.item())

            # Periodic evaluation
            if step % args.eval_every == 0 or step == args.steps - 1:
                val_loss = compute_val_loss(model, val_samples, args.device)
                val_losses.append(val_loss)

                if val_loss > best_val_loss:
                    consecutive_val_rises += 1
                else:
                    consecutive_val_rises = max(0, consecutive_val_rises - 1)
                    best_val_loss = val_loss

                if consecutive_val_rises >= 3 and not overfitting_warned:
                    print(f"\n  ⚠ Early stopping warning: val loss rose {consecutive_val_rises}× consecutively")
                    overfitting_warned = True

                hard_ppl_str = ""
                if step % args.ppl_every == 0 or step == args.steps - 1:
                    hard_ppl = compute_hard_ternary_ppl(
                        model, tokenizer, args.device,
                        num_sequences=args.val_samples_for_ppl, seq_len=args.seq_len
                    )
                    hard_ppls.append((step, hard_ppl))
                    hard_ppl_str = f"{hard_ppl:10.2f}"

                avg_delta = sum(
                    avg_param(m.delta) for m in model.modules()
                    if isinstance(m, SoftTernaryQuantizer)
                ) / max(n_quantizers, 1)
                avg_alpha = sum(
                    avg_param(m.alpha) for m in model.modules()
                    if isinstance(m, SoftTernaryQuantizer)
                ) / max(n_quantizers, 1)

                gap = val_loss - train_losses[-1]

                print(
                    f"  {step:5d} {tau:6.3f} {train_losses[-1]:10.4f} "
                    f"{val_loss:10.4f} {gap:8.4f} {avg_delta:8.4f} {avg_alpha:8.4f} "
                    f"{hard_ppl_str}"
                )
                sys.stdout.flush()

            step += 1

    # ── Final evaluation ───────────────────────────────────────
    print(f"\nFinal evaluation...")
    set_all_hard_mode(model, True)

    print("  Computing final hard ternary PPL (40 sequences, safe mode)...")
    final_ppl = evaluate_perplexity_safe(
        model, tokenizer, num_sequences=40, seq_len=args.seq_len, device=args.device
    )

    total_zero = 0
    total_weights = 0
    for m in model.modules():
        if isinstance(m, SoftTernaryQuantizer):
            w_q = m.get_quantized_weight()
            total_zero += (w_q == 0).sum().item()
            total_weights += w_q.numel()
    final_sparsity = total_zero / total_weights if total_weights > 0 else 0.0

    # ── Summary ────────────────────────────────────────────────
    improvement_vs_naive = (naive_ppl - final_ppl) / naive_ppl * 100 if naive_ppl > 0 else 0
    improvement_vs_stat = (stat_init_ppl - final_ppl) / stat_init_ppl * 100 if stat_init_ppl > 0 else 0
    gap_vs_fp16 = final_ppl / fp16_ppl if fp16_ppl > 0 else float("inf")

    final_train_loss = train_losses[-1] if train_losses else 0
    final_val_loss = val_losses[-1] if val_losses else 0
    train_val_gap = final_val_loss - final_train_loss

    go_criteria_1 = final_ppl < 5000
    go_criteria_2 = final_ppl < 500
    go_pass = go_criteria_1

    print()
    print("=" * 60)
    print("  AATQ Phase 1 Results — TinyLlama-1.1B")
    print("=" * 60)
    print(f"  FP16 Baseline PPL:           {fp16_ppl:10.2f}")
    print(f"  Naive Ternary PPL:           {naive_ppl:10.2f}")
    print(f"  Statistical Init PPL:        {stat_init_ppl:10.2f}")
    print(f"  Optimized Ternary PPL:       {final_ppl:10.2f}")
    print(f"  ─────────────────────────────────────")
    print(f"  Δ vs Naive:                  {improvement_vs_naive:+.1f}%")
    print(f"  Δ vs Stat Init:              {improvement_vs_stat:+.1f}%")
    print(f"  Optimized / FP16 ratio:      {gap_vs_fp16:.1f}x")
    print(f"  Final sparsity:              {final_sparsity:.1%}")
    print(f"  ─────────────────────────────────────")
    print(f"  Final train loss:            {final_train_loss:.4f}")
    print(f"  Final val loss:              {final_val_loss:.4f}")
    print(f"  Train/Val gap:               {train_val_gap:.4f}")
    print(f"  Consecutive val rises:       {consecutive_val_rises}")
    print(f"  Overfitting detected:        {'Yes' if overfitting_warned else 'No'}")
    print(f"  ─────────────────────────────────────")
    print(f"  Go/No-Go criteria:")
    print(f"    PPL < 5000 (10x better):   {'PASS' if go_criteria_1 else 'FAIL'}")
    print(f"    PPL < 500  (ideal):        {'PASS' if go_criteria_2 else 'NOT YET'}")
    print(f"  ─────────────────────────────────────")
    print(f"  Go/No-Go: {'PASS' if go_pass else 'FAIL — need more work'}")
    print("=" * 60)

    print(f"\n  Hard PPL progression:")
    for s, p in hard_ppls:
        print(f"    step {s:4d}: {p:.2f}")

    print(f"\n  Train loss (first 5):  {train_losses[:5]}")
    print(f"  Train loss (last 5):   {train_losses[-5:]}")
    if val_losses:
        print(f"  Val loss (first 3):    {val_losses[:3]}")
        print(f"  Val loss (last 3):     {val_losses[-3:]}")

    if teacher is not None:
        del teacher
        torch.cuda.empty_cache()

    print("\nDone.")


if __name__ == "__main__":
    main()
