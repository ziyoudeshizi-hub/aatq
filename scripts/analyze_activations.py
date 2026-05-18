#!/usr/bin/env python
"""Analyze how ternary quantization affects internal model representations.

Hooks into the residual stream output of each transformer layer, runs the
same inputs through FP16 and ternary (STE group-wise) versions, and computes
cosine similarity per layer.  Produces a JSON report and a console summary.

Usage:
    # Quick: statistical init only (no training, ~30s)
    python scripts/analyze_activations.py --mode quick

    # Full: STE training + comparison (default, ~8 min)
    python scripts/analyze_activations.py

    # Full with custom settings
    python scripts/analyze_activations.py --steps 300 --lr 0.02 --num-samples 20
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# -- project imports ----------------------------------------------------------
_CWD = Path(__file__).resolve().parent.parent
if str(_CWD) not in sys.path:
    sys.path.insert(0, str(_CWD))

from aatq.calibration import load_wikitext_samples, calibration_iterator
from aatq.quantize import (
    replace_with_soft_ternary,
    set_all_hard_mode,
    set_all_ste_mode,
    set_all_tau,
    collected_quantizer_params,
)
from aatq.soft_ternary import (
    SoftTernaryQuantizer,
    temperature_schedule,
)


# ═══════════════════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _layer_count(model: nn.Module) -> int:
    return sum(1 for _ in model.model.layers)


def _layer_output_hook(store: dict[int, torch.Tensor], layer_idx: int):
    """Capture residual-stream output of a single decoder layer."""
    def hook(_module, _inputs, output):
        # output[0] when layer returns a tuple (hidden_states, ...)
        hidden = output[0] if isinstance(output, tuple) else output
        store[layer_idx] = hidden.detach().cpu()
    return hook


def _embed_hook(store: dict[str, torch.Tensor]):
    """Capture embedding output (pre-first-layer)."""
    def hook(_module, _inputs, output):
        store["embed"] = output.detach().cpu()
    return hook


def _final_norm_hook(store: dict[str, torch.Tensor]):
    """Capture final LayerNorm output (post-last-layer)."""
    def hook(_module, _inputs, output):
        store["final_norm"] = output.detach().cpu()
    return hook


@torch.no_grad()
def collect_residual_activations(
    model: nn.Module,
    samples: list[dict],
    device: str,
) -> dict[str, torch.Tensor]:
    """Forward each sample through *model* and collect residual-stream states.

    Returns dict mapping:
        "embed"      → embedding output          [N, S, D]
        "layer_0"    → layer 0 residual output   [N, S, D]
        ...
        "layer_L-1"  → last layer residual output
        "final_norm" → final norm output          [N, S, D]
    """
    n_layers = _layer_count(model)
    per_sample: list[dict] = []

    model.eval()
    for sample in samples:
        store: dict = {}

        # register hooks
        handles = []
        handles.append(model.model.embed_tokens.register_forward_hook(_embed_hook(store)))
        for i, layer in enumerate(model.model.layers):
            handles.append(layer.register_forward_hook(_layer_output_hook(store, i)))
        handles.append(model.model.norm.register_forward_hook(_final_norm_hook(store)))

        input_ids = sample["input_ids"].unsqueeze(0).to(device)
        model(input_ids)

        per_sample.append(store)

        for h in handles:
            h.remove()

    # stack samples → dict of [N, S, D]
    keys = ["embed"] + [f"layer_{i}" for i in range(n_layers)] + ["final_norm"]
    stacked: dict[str, torch.Tensor] = {}
    for k in keys:
        stacked[k] = torch.stack([s[k] for s in per_sample])  # [N, S, D]
    return stacked


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two tensors, safe for fp16."""
    a_flat = a.reshape(-1).float()
    b_flat = b.reshape(-1).float()
    return float(F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0), dim=1)[0])


def l2_relative_error(fp16: torch.Tensor, ternary: torch.Tensor) -> float:
    """‖ternary − fp16‖₂ / ‖fp16‖₂"""
    f = fp16.reshape(-1).float()
    t = ternary.reshape(-1).float()
    return float((t - f).norm() / f.norm())


# ═══════════════════════════════════════════════════════════════════════════════
# STE training (inline, minimal — mirrors run_phase2_groupwise)
# ═══════════════════════════════════════════════════════════════════════════════

def train_ste(
    model: nn.Module,
    train_samples: list[dict],
    val_samples: list[dict],
    device: str,
    steps: int = 500,
    lr: float = 0.02,
    tau_start: float = 1.0,
    tau_end: float = 0.01,
    batch_size: int = 1,
    eval_every: int = 50,
    val_batches: int = 8,
):
    """Run STE group-wise training.  Returns final val loss."""
    set_all_ste_mode(model, True)
    optimizer = torch.optim.Adam(
        collected_quantizer_params(model), lr=lr,
        betas=(0.9, 0.99), eps=1e-4,
    )
    warmup_steps = max(1, int(steps * 0.1))

    train_batches = list(calibration_iterator(
        train_samples, batch_size=batch_size, device=device,
    ))
    batches_per_epoch = len(train_batches)
    epochs_needed = math.ceil(steps / batches_per_epoch)

    best_val = float("inf")
    step = 0

    for _epoch in range(epochs_needed):
        if step >= steps:
            break
        for batch in train_batches:
            if step >= steps:
                break

            tau = temperature_schedule(step, steps, tau_start, tau_end)
            set_all_tau(model, tau)

            if step < warmup_steps:
                lr_now = lr * (step + 1) / warmup_steps
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_now

            optimizer.zero_grad()
            outputs = model(**batch, labels=batch["input_ids"])
            loss = outputs.loss

            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad()
                step += 1
                continue

            loss.backward()

            quant_params = collected_quantizer_params(model)
            has_nan = any(
                p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
                for p in quant_params
            )
            if has_nan:
                optimizer.zero_grad()
                step += 1
                continue

            optimizer.step()

            # clamp
            with torch.no_grad():
                for m in model.modules():
                    if isinstance(m, SoftTernaryQuantizer):
                        m.delta.data.clamp_(min=1e-6, max=5.0)
                        m.alpha.data.clamp_(min=1e-6, max=5.0)

            del outputs

            if step % eval_every == 0 or step == steps - 1:
                # quick val loss
                model.eval()
                val_total = 0.0
                n_val = 0
                for vb in calibration_iterator(val_samples, batch_size=1, device=device):
                    vo = model(**vb, labels=vb["input_ids"])
                    val_total += vo.loss.item()
                    n_val += 1
                    if n_val >= val_batches:
                        break
                model.train()
                val_loss = val_total / max(n_val, 1)
                if val_loss < best_val:
                    best_val = val_loss

            step += 1

    set_all_ste_mode(model, False)
    set_all_hard_mode(model, True)
    return best_val


# ═══════════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="AATQ activation similarity analysis")
    p.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num-samples", type=int, default=20,
                   help="Number of test samples for activation comparison")
    p.add_argument("--seq-len", type=int, default=256,
                   help="Sequence length for samples")
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--steps", type=int, default=500,
                   help="STE training steps (ignored in --mode quick)")
    p.add_argument("--lr", type=float, default=0.02,
                   help="Learning rate for STE training")
    p.add_argument("--cal-samples", type=int, default=128,
                   help="Calibration samples for STE training")
    p.add_argument("--mode", type=str, default="full",
                   choices=["quick", "full"],
                   help="quick = statistical init only; full = STE training first")
    p.add_argument("--output", type=str, default=None,
                   help="Output JSON path (default: activation_sim_<model_tag>.json)")
    p.add_argument("--plot", type=str, default=None,
                   help="Save matplotlib chart to this path (e.g. activation_sim.png)")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device

    tag = args.model.replace("/", "_")
    out_json = args.output or f"activation_sim_{tag}_gs{args.group_size}.json"

    print("=" * 64)
    print("  AATQ — Activation Similarity Analysis")
    print(f"  Model:       {args.model}")
    print(f"  Mode:        {args.mode}")
    print(f"  Samples:     {args.num_samples} × seq={args.seq_len}")
    if args.mode == "full":
        print(f"  STE:         {args.steps} steps, lr={args.lr}, gs={args.group_size}")
    print("=" * 64)

    # ── 1. Load model & tokenizer ──────────────────────────────────────────
    print("\n[1/5] Loading FP16 model …")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    n_layers = _layer_count(model)
    print(f"  Layers: {n_layers}")

    # ── 2. Load test samples (held-out, NOT calibration) ───────────────────
    print("\n[2/5] Loading test samples …")
    test_samples = load_wikitext_samples(
        tokenizer, split="test", num_samples=args.num_samples,
        seq_len=args.seq_len, seed=1234,
    )
    print(f"  {len(test_samples)} test samples ready")

    # ── 3. Collect FP16 activations ────────────────────────────────────────
    print("\n[3/5] Collecting FP16 activations …")
    t0 = time.perf_counter()
    fp16_acts = collect_residual_activations(model, test_samples, device)
    t_fp16 = time.perf_counter() - t0
    print(f"  Done in {t_fp16:.1f}s")

    # ── 4. Apply ternary & optionally train ────────────────────────────────
    print("\n[4/5] Applying ternary quantization …")
    skip_names = {"lm_head", "embed_tokens"}
    model = replace_with_soft_ternary(model, skip_names=skip_names, group_size=args.group_size)

    if args.mode == "quick":
        print("  Using statistical init (no training)")
        set_all_hard_mode(model, True)
    else:
        print(f"  STE training ({args.steps} steps, lr={args.lr}) …")
        cal_samples = load_wikitext_samples(
            tokenizer, split="train", num_samples=args.cal_samples,
            seq_len=args.seq_len, seed=42,
        )
        n_train = int(len(cal_samples) * 0.8)
        train_samples, val_samples = cal_samples[:n_train], cal_samples[n_train:]
        print(f"  Train: {len(train_samples)}, Val: {len(val_samples)}")

        t_train = time.perf_counter()
        best_val = train_ste(
            model, train_samples, val_samples, device,
            steps=args.steps, lr=args.lr, batch_size=1,
        )
        elapsed = time.perf_counter() - t_train
        print(f"  Done in {elapsed:.0f}s  —  best val loss: {best_val:.4f}")

    # ── 5. Collect ternary activations ─────────────────────────────────────
    print("\n[5/5] Collecting ternary activations …")
    t0 = time.perf_counter()
    ternary_acts = collect_residual_activations(model, test_samples, device)
    t_tern = time.perf_counter() - t0
    print(f"  Done in {t_tern:.1f}s")

    # ── Compute similarities ───────────────────────────────────────────────
    print("\n" + "─" * 64)
    print("  Results")
    print("─" * 64)

    keys = ["embed"] + [f"layer_{i}" for i in range(n_layers)] + ["final_norm"]

    results: list[dict] = []
    header = f"  {'Hook':<14s} {'cos_sim':>8s}  {'L2_rel':>8s}  {'degradation':>12s}"
    print(header)
    print(f"  {'-'*14} {'-'*8}  {'-'*8}  {'-'*12}")

    for k in keys:
        cs = cosine_sim(fp16_acts[k], ternary_acts[k])
        l2 = l2_relative_error(fp16_acts[k], ternary_acts[k])
        deg = 1.0 - cs  # degradation = 1 - cosine similarity
        print(f"  {k:<14s} {cs:8.4f}  {l2:8.4f}  {deg:12.6f}")
        results.append({"hook": k, "cosine_sim": cs, "l2_rel_err": l2, "degradation": deg})

    # ── Summary stats ──────────────────────────────────────────────────────
    layer_cos = [r["cosine_sim"] for r in results if r["hook"].startswith("layer_")]
    avg_cos = sum(layer_cos) / len(layer_cos)
    min_cos = min(layer_cos)
    min_layer = results[1:][layer_cos.index(min_cos)]["hook"]  # +1 to skip embed
    max_cos = max(layer_cos)
    max_layer = results[1:][layer_cos.index(max_cos)]["hook"]

    print(f"\n  ── Summary ──")
    print(f"  Avg layer cosine sim:  {avg_cos:.4f}")
    print(f"  Best layer:            {max_layer} ({max_cos:.4f})")
    print(f"  Worst layer:           {min_layer} ({min_cos:.4f})")
    print(f"  Embedding cos sim:     {results[0]['cosine_sim']:.4f}")
    print(f"  Final norm cos sim:    {results[-1]['cosine_sim']:.4f}")

    # ── Sparsity ───────────────────────────────────────────────────────────
    total_zero = 0
    total_weights = 0
    for m in model.modules():
        if isinstance(m, SoftTernaryQuantizer):
            w_q = m.get_quantized_weight()
            total_zero += (w_q == 0).sum().item()
            total_weights += w_q.numel()
    sparsity = total_zero / total_weights if total_weights > 0 else 0.0
    print(f"  Ternary sparsity:      {sparsity:.1%}")

    # ── Interpretation ─────────────────────────────────────────────────────
    print(f"\n  ── Interpretation ──")
    if min_cos > 0.95:
        print(f"  Activations highly preserved across all layers (>0.95).")
        print(f"  Ternary quantization barely disturbs internal representations.")
    elif min_cos > 0.80:
        print(f"  Activations well preserved (>0.80). Deep layers show more drift.")
    elif min_cos > 0.60:
        print(f"  Moderate activation degradation in deeper layers.")
        print(f"  Consistent with accumulated quantization error through depth.")
    else:
        print(f"  Significant activation deviation in some layers.")
        print(f"  Consider mixed-precision: keep critical layers at higher precision.")

    # ── Write JSON ─────────────────────────────────────────────────────────
    report = {
        "model": args.model,
        "mode": args.mode,
        "group_size": args.group_size,
        "n_layers": n_layers,
        "num_test_samples": args.num_samples,
        "seq_len": args.seq_len,
        "avg_layer_cosine_sim": avg_cos,
        "min_layer_cosine_sim": min_cos,
        "min_layer": min_layer,
        "max_layer_cosine_sim": max_cos,
        "max_layer": max_layer,
        "embed_cosine_sim": results[0]["cosine_sim"],
        "final_norm_cosine_sim": results[-1]["cosine_sim"],
        "sparsity": sparsity,
        "per_hook": results,
    }
    if args.mode == "full":
        report["ste_steps"] = args.steps
        report["ste_lr"] = args.lr
        report["cal_samples"] = args.cal_samples

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  Report → {out_json}")

    # ── Optional plot ──────────────────────────────────────────────────────
    if args.plot:
        _make_plot(results, n_layers, args.plot, args.model, sparsity, args.mode)
        print(f"  Chart  → {args.plot}")

    print("\nDone.")


# ═══════════════════════════════════════════════════════════════════════════════
# matplotlib chart (optional)
# ═══════════════════════════════════════════════════════════════════════════════

def _make_plot(
    results: list[dict],
    n_layers: int,
    path: str,
    model_name: str,
    sparsity: float,
    mode: str,
):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not available, skipping chart)")
        return

    layer_idx = list(range(n_layers))
    cos_sims = [r["cosine_sim"] for r in results if r["hook"].startswith("layer_")]
    l2_errs = [r["l2_rel_err"] for r in results if r["hook"].startswith("layer_")]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # top: cosine similarity
    ax1.plot(layer_idx, cos_sims, "b-o", markersize=4, linewidth=1.5)
    ax1.axhline(y=1.0, color="gray", linestyle="--", alpha=0.4)
    ax1.set_ylabel("Cosine Similarity")
    ax1.set_title(f"FP16 vs Ternary Activation Similarity — {model_name}\n"
                  f"mode={mode}, sparsity={sparsity:.1%}")
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)

    # bottom: L2 relative error
    ax2.plot(layer_idx, l2_errs, "r-o", markersize=4, linewidth=1.5)
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("L2 Relative Error")
    ax2.grid(True, alpha=0.3)

    # shade deep layers
    mid = n_layers // 2
    ax1.axvspan(mid, n_layers - 1, alpha=0.06, color="red")
    ax2.axvspan(mid, n_layers - 1, alpha=0.06, color="red")

    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
