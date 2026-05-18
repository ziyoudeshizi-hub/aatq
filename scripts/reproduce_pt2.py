#!/usr/bin/env python
"""Reproduce PT²-LLM simplified baseline on a small model.

This script doesn't attempt full reproduction — it focuses on:
1. Loading a model and applying ITF-like ternary quantization
2. Observing what happens when Δ is optimized without protection

Goal: confirm PT²-LLM's claim that optimizing Δ without safeguards
leads to overfitting on the calibration set.
"""

import argparse
import copy
import sys

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from aatq.calibration import calibration_iterator, load_wikitext_samples
from aatq.eval import evaluate_perplexity_fast
from aatq.quantize import replace_with_soft_ternary, set_all_tau, set_all_hard_mode
from aatq.utils import count_quantizable_layers


def parse_args():
    parser = argparse.ArgumentParser(description="Reproduce PT²-LLM Δ overfitting")
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-cal-samples", type=int, default=128)
    parser.add_argument("--cal-batch-size", type=int, default=4)
    parser.add_argument("--optim-steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--tau-start", type=float, default=1.0)
    parser.add_argument("--tau-end", type=float, default=0.01)
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 50)
    print("  PT²-LLM Δ Overfitting Reproduction")
    print(f"  Model: {args.model}")
    print("=" * 50)

    # ── Load model ────────────────────────────────────────────
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

    num_layers = count_quantizable_layers(model)
    print(f"  Quantizable layers: {num_layers}")

    # ── Baseline FP16 perplexity ──────────────────────────────
    print("\n[2/6] FP16 baseline perplexity...")
    fp16_ppl = evaluate_perplexity_fast(model, tokenizer, num_sequences=40, device=args.device)
    print(f"  FP16 PPL: {fp16_ppl:.2f}")

    # ── Replace with soft ternary quantizers ──────────────────
    print("\n[3/6] Replacing Linear layers with SoftTernaryQuantizer...")
    model = replace_with_soft_ternary(model)
    set_all_tau(model, args.tau_start)
    set_all_hard_mode(model, False)

    # ── Load calibration data ─────────────────────────────────
    print("\n[4/6] Loading calibration data...")
    cal_samples = load_wikitext_samples(
        tokenizer, split="train", num_samples=args.num_cal_samples, seq_len=2048
    )
    cal_loader = calibration_iterator(cal_samples, batch_size=args.cal_batch_size, device=args.device)

    # ── Create optimizer, only Δ and α params ─────────────────
    quant_params = []
    for m in model.modules():
        if hasattr(m, "delta") and isinstance(m.delta, nn.Parameter):
            quant_params.append(m.delta)
        if hasattr(m, "alpha") and isinstance(m.alpha, nn.Parameter):
            quant_params.append(m.alpha)
    print(f"  Optimizable params: {sum(p.numel() for p in quant_params)}")
    optimizer = torch.optim.AdamW(quant_params, lr=args.lr)

    # ── Record ppl every N steps ──────────────────────────────
    ppl_log = []
    eval_every = max(args.optim_steps // 20, 5)

    print(f"\n[5/6] Optimizing (no distillation, no mixup) — observe overfitting...")
    step = 0
    cal_entries = list(cal_loader)

    for epoch in range(3):
        for batch in cal_entries:
            tau = args.tau_end + (args.tau_start - args.tau_end) * (1 - step / args.optim_steps)
            set_all_tau(model, tau)

            optimizer.zero_grad()
            outputs = model(**batch, labels=batch["input_ids"])
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            if step % eval_every == 0:
                set_all_hard_mode(model, True)
                ppl = evaluate_perplexity_fast(model, tokenizer, num_sequences=20, device=args.device)
                set_all_hard_mode(model, False)
                ppl_log.append((step, ppl))
                print(f"  Step {step:4d}: τ={tau:.3f}, loss={loss.item():.3f}, PPL={ppl:.2f}")

            step += 1
            if step >= args.optim_steps:
                break
        if step >= args.optim_steps:
            break

    # ── Final evaluation ──────────────────────────────────────
    print("\n[6/6] Final evaluation...")
    set_all_hard_mode(model, True)

    cal_ppl = evaluate_perplexity_fast(model, tokenizer, num_sequences=40, device=args.device)
    print(f"\n  FP16 PPL  : {fp16_ppl:.2f}")
    print(f"  Final PPL : {cal_ppl:.2f}")
    print(f"  Δ PPL     : {cal_ppl - fp16_ppl:.2f}")

    print("\n  PPL progression:")
    for s, p in ppl_log:
        print(f"    step {s:4d}: {p:.2f}")

    print("\nDone. Check if PPL degraded during optimization (overfitting signal).")


if __name__ == "__main__":
    main()
