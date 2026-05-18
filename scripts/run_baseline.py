#!/usr/bin/env python
"""Baseline: naive ternary quantization on TinyLlama-1.1B.

Applies pure statistical ternary quantization (Δ = 0.7σ, α = mean(|w| | w>Δ))
to all Linear layers, then evaluates perplexity on wikitext-2.

This establishes the lower bound to beat. Our Soft Ternary + annealing
approach must produce better perplexity than this naive baseline.
"""

import argparse
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from aatq.eval import evaluate_perplexity_fast
from aatq.quantize import apply_naive_ternary
from aatq.utils import count_quantizable_layers, model_size_mb


def parse_args():
    parser = argparse.ArgumentParser(description="Naive ternary quantization baseline")
    parser.add_argument(
        "--model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Model name on HuggingFace",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Target device",
    )
    parser.add_argument(
        "--num-eval-sequences",
        type=int,
        default=40,
        help="Number of sequences for perplexity evaluation",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dtype = getattr(torch, args.dtype)

    print(f"========================================")
    print(f"  AATQ — Naive Ternary Baseline")
    print(f"  Model: {args.model}")
    print(f"  Device: {args.device}, dtype: {args.dtype}")
    print(f"========================================")

    # ── Load model ────────────────────────────────────────────
    print("\n[1/5] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()

    num_layers = count_quantizable_layers(model)
    fp16_size = model_size_mb(model, dtype_bytes=2)
    print(f"  Quantizable Linear layers: {num_layers}")
    print(f"  Model size (FP16): {fp16_size:.1f} MB")

    # ── Baseline: FP16 perplexity ─────────────────────────────
    print("\n[2/5] Evaluating FP16 baseline perplexity...")
    t0 = time.time()
    fp16_ppl = evaluate_perplexity_fast(
        model,
        tokenizer,
        num_sequences=args.num_eval_sequences,
        device=args.device,
    )
    fp16_time = time.time() - t0
    print(f"  FP16 perplexity: {fp16_ppl:.2f}")
    print(f"  Eval time: {fp16_time:.1f}s")
    sys.stdout.flush()

    # ── Apply naive ternary quantization ─────────────────────
    print("\n[3/5] Applying naive ternary quantization...")
    model = apply_naive_ternary(model)

    # ── Evaluate ternary perplexity ──────────────────────────
    print("\n[4/5] Evaluating ternary perplexity...")
    t0 = time.time()
    ternary_ppl = evaluate_perplexity_fast(
        model,
        tokenizer,
        num_sequences=args.num_eval_sequences,
        device=args.device,
    )
    ternary_time = time.time() - t0
    print(f"  Ternary perplexity: {ternary_ppl:.2f}")
    print(f"  Eval time: {ternary_time:.1f}s")
    sys.stdout.flush()

    # ── Summary ──────────────────────────────────────────────
    print("\n[5/5] Results")
    print(f"========================================")
    print(f"  FP16 PPL:      {fp16_ppl:8.2f}")
    print(f"  Ternary PPL:   {ternary_ppl:8.2f}")
    print(f"  Δ PPL:         {ternary_ppl - fp16_ppl:8.2f}")
    print(f"  Degradation:   {(ternary_ppl / fp16_ppl - 1) * 100:.1f}%")
    print(f"  Model size:    {fp16_size:.1f} MB → {fp16_size * (2 * 2 / 16):.1f} MB (2-bit naive)")
    print(f"========================================")
    print(f"\n(Soft Ternary + annealing must beat {ternary_ppl:.2f})")


if __name__ == "__main__":
    main()
