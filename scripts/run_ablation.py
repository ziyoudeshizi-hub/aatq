#!/usr/bin/env python
"""AATQ Ablation Experiments for Paper Revision.

Two experiments:
  1. N=32 samples for layer analysis (with std err) — addresses reviewer concern on N=4
  2. L0-L2 FP16 ablation: keep first K layers in FP16, quantize rest — proves causality

Usage:
    python run_ablation.py --model qwen2-7b
    python run_ablation.py --model qwen2-7b --skip-full  # only run ablation, skip N=32 re-analysis
"""

from __future__ import annotations
import argparse, json, os, sys, time
from datetime import datetime, timezone, timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Import from main script ──────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_7b_all import (
    MODELS, evaluate_ppl, make_calibration_data,
    quantize_model_itf_sequential, collect_block_hidden_states
)


def compute_layer_similarity_with_stats(fp16_states, ternary_states):
    """Like compute_layer_similarity but returns per-sample stats for std err."""
    n_layers = len(fp16_states)
    layer_metrics = []

    for i in range(n_layers):
        fp = fp16_states[i].float()   # (N, 1, seq_len, d_model)
        tq = ternary_states[i].float()

        # Per-sample cosine (across all tokens in each sample)
        per_sample_cos = []
        for s in range(fp.shape[0]):
            fp_flat = fp[s].flatten(0, -2)  # (seq_len, d_model)
            tq_flat = tq[s].flatten(0, -2)
            cos = F.cosine_similarity(fp_flat, tq_flat, dim=-1)
            per_sample_cos.append(cos.mean().item())

        cos_array = np.array(per_sample_cos)
        # Also compute token-level stats across all samples
        fp_all = fp.flatten(0, -2)
        tq_all = tq.flatten(0, -2)
        cos_all = F.cosine_similarity(fp_all, tq_all, dim=-1)
        mse = (fp_all - tq_all).pow(2).mean().item()
        fp_var = fp_all.var(dim=-1).mean().item()

        layer_metrics.append({
            "layer": i,
            "cosine_mean": round(cos_array.mean(), 6),
            "cosine_std": round(cos_array.std(), 6),
            "cosine_stderr": round(cos_array.std() / np.sqrt(len(cos_array)), 6),
            "cosine_median": round(float(np.median(cos_array)), 6),
            "cosine_min_token": round(cos_all.min().item(), 6),
            "cosine_max_token": round(cos_all.max().item(), 6),
            "n_samples": len(cos_array),
            "mse": round(mse, 6),
            "rel_mse": round(mse / max(fp_var, 1e-10), 4),
        })

    return layer_metrics


def quantize_model_partial(model, skip_first_k=0, skip_last_k=0, group_size=128, n_iter=20, calibration_data=None):
    """Quantize model but keep first K and last K blocks in FP16.
    
    This is the key ablation: if error cascading causes L3 collapse,
    keeping L0-L2 in FP16 should prevent it.
    """
    blocks = model.model.layers
    n_blocks = len(blocks)
    print(f"  Partial quantize: skip first {skip_first_k}, skip last {skip_last_k}")
    print(f"  Quantizing blocks {skip_first_k} to {n_blocks - skip_last_k - 1}")

    # We need calibration hidden states up to the first quantized block
    # Collect hidden states for each block sequentially
    hidden_list = []
    with torch.no_grad():
        for sample in calibration_data:
            # Get embeddings
            embed = model.model.embed_tokens(sample)
            hidden_list.append(embed)

    # Process blocks sequentially
    from run_7b_all import itf_quantize_weight_fast

    total_mse = 0.0
    total_elements = 0
    n_quantized_blocks = 0

    for block_idx in range(n_blocks):
        block = blocks[block_idx]

        if block_idx < skip_first_k or block_idx >= (n_blocks - skip_last_k):
            # Keep this block in FP16 — just pass hidden states through
            new_hidden = []
            with torch.no_grad():
                for h in hidden_list:
                    out = block(h.cuda() if not h.is_cuda else h)
                    result = out[0] if isinstance(out, tuple) else out
                    new_hidden.append(result)
            hidden_list = new_hidden
            continue

        # Quantize this block
        n_quantized_blocks += 1
        linears = [(n, m) for n, m in block.named_modules()
                   if isinstance(m, nn.Linear) and m.weight.shape[0] >= 64
                   and m.weight.shape[1] % group_size == 0]

        # Collect activation stats
        activation_stats = {}
        hooks = []
        def make_hook(name):
            def hook_fn(module, input, output):
                x = input[0]
                if x.dim() == 3:
                    x = x.reshape(-1, x.shape[-1])
                col_sq = (x.float() ** 2).sum(dim=0).cpu()
                if name in activation_stats:
                    activation_stats[name] += col_sq
                else:
                    activation_stats[name] = col_sq
            return hook_fn

        for name, module in linears:
            hooks.append(module.register_forward_hook(make_hook(name)))

        with torch.no_grad():
            for h in hidden_list:
                block(h.cuda() if not h.is_cuda else h)

        for h in hooks:
            h.remove()

        # Quantize layers in this block
        for name, module in linears:
            w = module.weight.data.float().cpu().numpy()
            O, I = w.shape
            col_weights = None
            if name in activation_stats:
                col_weights = np.sqrt(activation_stats[name].numpy() + 1e-10).astype(np.float32)

            w_q = itf_quantize_weight_fast(w, group_size=group_size, n_iter=n_iter, col_weights=col_weights)
            mse = ((w - w_q) ** 2).mean()
            total_mse += mse * O * I
            total_elements += O * I
            module.weight.data = torch.from_numpy(w_q).to(module.weight.dtype).to(module.weight.device)

        # Forward through quantized block
        new_hidden = []
        with torch.no_grad():
            for h in hidden_list:
                out = block(h.cuda() if not h.is_cuda else h)
                result = out[0] if isinstance(out, tuple) else out
                new_hidden.append(result)
        hidden_list = new_hidden

        if (n_quantized_blocks) % 4 == 0:
            avg_mse = total_mse / max(total_elements, 1)
            print(f"  [Block {block_idx+1}/{n_blocks}] quantized={n_quantized_blocks}, avg_mse={avg_mse:.6f}")

    avg_mse = total_mse / max(total_elements, 1)
    print(f"  Done: {n_quantized_blocks} blocks quantized, avg_mse={avg_mse:.6f}")
    return model, avg_mse


def parse_args():
    p = argparse.ArgumentParser(description="AATQ Ablation Experiments")
    p.add_argument("--model", default="qwen2-7b", choices=list(MODELS))
    p.add_argument("--gs", type=int, default=128)
    p.add_argument("--itf-iter", type=int, default=20)
    p.add_argument("--cal-samples", type=int, default=128)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--analyze-samples", type=int, default=32, help="Samples for layer analysis (default 32)")
    p.add_argument("--output-dir", default="./results_7b")
    p.add_argument("--skip-full", action="store_true", help="Skip full N=32 reanalysis, only do ablation")
    return p.parse_args()


def main():
    args = parse_args()
    model_id = MODELS[args.model]
    os.makedirs(args.output_dir, exist_ok=True)
    tz = timezone(timedelta(hours=8))

    print("=" * 70)
    print(f"  AATQ Ablation Experiments")
    print(f"  Model:  {model_id}")
    print(f"  GPU:    {torch.cuda.get_device_name(0)}")
    print(f"  Time:   {datetime.now(tz).isoformat()}")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    cal_data = make_calibration_data(tokenizer, n_samples=args.cal_samples, seq_len=args.seq_len)
    results = {"model": model_id, "start_time": datetime.now(tz).isoformat()}

    # ══════════════════════════════════════════════════════════════════════════
    # Experiment 1: N=32 Full Analysis (with std err)
    # ══════════════════════════════════════════════════════════════════════════
    if not args.skip_full:
        print("\n" + "=" * 70)
        print(f"  EXPERIMENT 1: Full layer analysis with N={args.analyze_samples} (std err)")
        print("=" * 70)

        # Load fresh model, collect FP16 states
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
        model.eval()

        analyze_data = cal_data[:args.analyze_samples]
        print(f"\n  Collecting FP16 hidden states ({args.analyze_samples} samples)...")
        fp16_states = collect_block_hidden_states(model, analyze_data)
        print(f"  FP16 states: {len(fp16_states)} layers, shape={fp16_states[0].shape}")

        # Quantize
        print(f"\n  Quantizing (ITF+AGA, gs={args.gs}, iter={args.itf_iter})...")
        t0 = time.time()
        model, avg_mse = quantize_model_itf_sequential(
            model, group_size=args.gs, n_iter=args.itf_iter, calibration_data=cal_data)
        quant_time = time.time() - t0
        print(f"  Quantization done: {quant_time:.0f}s, avg_mse={avg_mse:.8f}")

        # Collect ternary states
        print(f"\n  Collecting ternary hidden states ({args.analyze_samples} samples)...")
        ternary_states = collect_block_hidden_states(model, analyze_data)

        # Compute with stats
        print("  Computing per-layer similarity with std err...")
        layer_metrics = compute_layer_similarity_with_stats(fp16_states, ternary_states)
        del fp16_states, ternary_states
        torch.cuda.empty_cache()

        # Summary
        cos_means = [m["cosine_mean"] for m in layer_metrics]
        cos_stderrs = [m["cosine_stderr"] for m in layer_metrics]
        worst = min(range(len(cos_means)), key=lambda i: cos_means[i])
        print(f"\n  Results (N={args.analyze_samples}):")
        print(f"    Worst layer: L{worst} cos={cos_means[worst]:.4f} +/- {cos_stderrs[worst]:.4f}")
        print(f"    L0: cos={cos_means[0]:.4f} +/- {cos_stderrs[0]:.4f}")
        print(f"    L3: cos={cos_means[3]:.4f} +/- {cos_stderrs[3]:.4f}")

        results["full_analysis_n32"] = {
            "n_samples": args.analyze_samples,
            "quant_time_s": round(quant_time, 1),
            "avg_mse": round(float(avg_mse), 8),
            "layer_metrics": layer_metrics,
        }

        del model
        torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════════════════
    # Experiment 2: L0-L2 FP16 Ablation
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  EXPERIMENT 2: Causality Ablation (keep L0-L2 in FP16)")
    print("=" * 70)

    ablation_configs = [
        {"skip_first": 0, "name": "all_quantized"},
        {"skip_first": 3, "name": "skip_first_3"},
        {"skip_first": 5, "name": "skip_first_5"},
    ]

    results["ablation"] = {}
    analyze_data = cal_data[:min(args.analyze_samples, 8)]  # Use 8 for ablation (faster)

    for cfg in ablation_configs:
        print(f"\n  --- {cfg['name']}: skip_first={cfg['skip_first']} ---")

        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
        model.eval()

        # Collect FP16 states
        fp16_states = collect_block_hidden_states(model, analyze_data)

        # Quantize with partial skip
        if cfg["skip_first"] == 0:
            model, avg_mse = quantize_model_itf_sequential(
                model, group_size=args.gs, n_iter=args.itf_iter, calibration_data=cal_data)
        else:
            model, avg_mse = quantize_model_partial(
                model, skip_first_k=cfg["skip_first"], group_size=args.gs,
                n_iter=args.itf_iter, calibration_data=cal_data)

        # Evaluate PPL
        ppl = evaluate_ppl(model, tokenizer)
        print(f"  PPL = {ppl:.2f}")

        # Collect ternary states and compare
        ternary_states = collect_block_hidden_states(model, analyze_data)
        layer_metrics = compute_layer_similarity_with_stats(fp16_states, ternary_states)
        del fp16_states, ternary_states

        # Key layer cosines
        cos_L3 = layer_metrics[3]["cosine_mean"] if len(layer_metrics) > 3 else None
        cos_L5 = layer_metrics[5]["cosine_mean"] if len(layer_metrics) > 5 else None
        cos_L10 = layer_metrics[10]["cosine_mean"] if len(layer_metrics) > 10 else None

        print(f"  L3 cos={cos_L3:.4f}, L5 cos={cos_L5:.4f}, L10 cos={cos_L10:.4f}")

        results["ablation"][cfg["name"]] = {
            "skip_first": cfg["skip_first"],
            "ppl": round(ppl, 2),
            "avg_mse": round(float(avg_mse), 8),
            "layer_metrics": layer_metrics,
        }

        del model
        torch.cuda.empty_cache()

    # ── Save ──────────────────────────────────────────────────────────────────
    results["end_time"] = datetime.now(tz).isoformat()
    out_path = os.path.join(args.output_dir, f"{args.model}_ablation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to: {out_path}")
    print("  Done!")


if __name__ == "__main__":
    main()
