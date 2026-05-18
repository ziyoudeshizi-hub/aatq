#!/usr/bin/env python
"""AATQ 7B Full Experiment Suite — runs all methods on 7B for paper.

This script runs:
  1. FP16 baseline PPL
  2. STE (group-wise, 500 steps) → PPL
  3. ITF + AGA (sequential) → PPL
  4. Save all results to JSON

Usage:
    python scripts/run_7b_all.py                        # Default: Qwen2.5-7B
    python scripts/run_7b_all.py --model llama2-7b      # LLaMA-2-7B
    python scripts/run_7b_all.py --quick                # Quick test (fewer steps)
"""

from __future__ import annotations
import argparse, json, math, os, sys, time
from datetime import datetime, timezone, timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Model Registry ───────────────────────────────────────────────────────────
MODELS = {
    "qwen2-7b":   "Qwen/Qwen2.5-7B",
    "llama2-7b":  "meta-llama/Llama-2-7b-hf",
    "llama3-8b":  "meta-llama/Meta-Llama-3-8B",
    "tinyllama":  "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}


# ══════════════════════════════════════════════════════════════════════════════
#  METHOD 1: STE (Straight-Through Estimator)
# ══════════════════════════════════════════════════════════════════════════════

class STQ_Group(nn.Module):
    """Group-wise Soft Ternary Quantizer for STE training."""

    def __init__(self, linear: nn.Linear, group_size: int = 128):
        super().__init__()
        w = linear.weight.data.float()
        O, I = w.shape
        assert I % group_size == 0
        self.gs = group_size
        self.G = I // group_size
        self.O, self.I = O, I
        self.register_buffer("orig_w", w.bfloat16())
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
        delta_3d = self.delta.bfloat16().unsqueeze(-1)
        alpha_3d = self.alpha.bfloat16().unsqueeze(-1)
        abs_w = w_3d.abs()
        sgn_w = w_3d.sign()
        if self.hard:
            mask = (abs_w > delta_3d).bfloat16()
        else:
            mask_hard = (abs_w > delta_3d).bfloat16()
            sig_in = ((abs_w - delta_3d) / max(self.tau, 0.25)).clamp(-10, 10)
            mask_soft = torch.sigmoid(sig_in)
            mask = mask_hard.detach() + mask_soft - mask_soft.detach()
        wq = (alpha_3d * mask * sgn_w).view(self.O, self.I)
        return F.linear(x, wq, self.bias.bfloat16() if self.bias is not None else None)

    def get_hard_weight(self):
        w_3d = self.orig_w.view(self.O, self.G, self.gs)
        delta_3d = self.delta.detach().bfloat16().unsqueeze(-1)
        alpha_3d = self.alpha.detach().bfloat16().unsqueeze(-1)
        mask = (w_3d.abs() > delta_3d).bfloat16()
        return (alpha_3d * mask * w_3d.sign()).view(self.O, self.I)


def apply_ste_quantization(model, group_size=128):
    """Replace Linear layers with STQ_Group for STE training."""
    skip_names = {"lm_head", "embed_tokens", "rotary_emb"}
    count, params = 0, 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear) or module.weight.shape[0] < 64:
            continue
        if module.weight.shape[1] % group_size != 0:
            continue
        if name.split(".")[-1] in skip_names:
            continue
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        q = STQ_Group(module, group_size)
        setattr(parent, parts[-1], q)
        count += 1
        params += q.delta.numel() + q.alpha.numel()
    print(f"  STE: {count} layers quantized, {params:,} trainable params")
    return model, params


def train_ste(model, batches, steps=500, lr=0.01):
    """Train STE with cosine tau schedule."""
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
        progress = step / max(steps - 1, 1)
        tau = 0.25 + 0.75 * 0.5 * (1 + math.cos(math.pi * progress))
        for m in model.modules():
            if isinstance(m, STQ_Group):
                m.tau = tau

        batch = batches[step % len(batches)]
        out = model(batch, labels=batch)

        if torch.isnan(out.loss) or torch.isinf(out.loss):
            print(f"  NaN at step {step}, restoring best")
            if best_state:
                _restore(model, best_state)
            break

        opt.zero_grad()
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        if step < 20:
            for pg in opt.param_groups:
                pg["lr"] = lr * (step + 1) / 20
        opt.step()

        loss_val = out.loss.item()
        if loss_val < best_loss:
            best_loss = loss_val
            best_state = _snapshot(model)

        if step % 100 == 0 or step == steps - 1:
            print(f"  STE {step:4d}/{steps} loss={loss_val:.4f} tau={tau:.3f} [{time.time()-t0:.0f}s]")

    model.gradient_checkpointing_disable()
    if best_state:
        _restore(model, best_state)
    return model, best_loss


def freeze_ste(model):
    """Convert STQ_Group back to hard ternary Linear."""
    for name, module in list(model.named_modules()):
        if isinstance(module, STQ_Group):
            w_q = module.get_hard_weight()
            new_linear = nn.Linear(module.I, module.O, bias=module.bias is not None,
                                   device="cuda", dtype=torch.bfloat16)
            new_linear.weight.data = w_q
            if module.bias is not None:
                new_linear.bias.data = module.bias.data.bfloat16()
            parts = name.split(".")
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], new_linear)
    return model


def _snapshot(model):
    state = {}
    for m in model.modules():
        if isinstance(m, STQ_Group):
            state[id(m)] = {"delta": m.delta.data.cpu().clone(), "alpha": m.alpha.data.cpu().clone()}
    return state


def _restore(model, state):
    for m in model.modules():
        if isinstance(m, STQ_Group):
            if id(m) in state:
                m.delta.data.copy_(state[id(m)]["delta"])
                m.alpha.data.copy_(state[id(m)]["alpha"])


# ══════════════════════════════════════════════════════════════════════════════
#  METHOD 2: ITF + AGA (Sequential)
# ══════════════════════════════════════════════════════════════════════════════

def itf_quantize_weight_fast(weight, group_size=128, n_iter=20, col_weights=None):
    """Vectorized ITF: non-symmetric ternary {mu-alpha, mu, mu+alpha} per group."""
    O, I = weight.shape
    assert I % group_size == 0
    n_groups = I // group_size

    w_3d = weight.reshape(O, n_groups, group_size)
    w_flat = w_3d.reshape(-1, group_size).astype(np.float64)
    N = w_flat.shape[0]

    if col_weights is not None:
        cw_groups = col_weights.reshape(n_groups, group_size).astype(np.float64)
        cw_flat = np.tile(cw_groups, (O, 1))
        cw_mean = cw_flat.mean(axis=1, keepdims=True)
        cw_mean = np.maximum(cw_mean, 1e-10)
        cw_flat = cw_flat / cw_mean
    else:
        cw_flat = np.ones((N, group_size), dtype=np.float64)

    mu = np.median(w_flat, axis=1)
    alpha = np.std(np.abs(w_flat - mu[:, None]), axis=1) * 0.8
    alpha = np.maximum(alpha, 1e-10)

    for _ in range(n_iter):
        val_neg = (mu - alpha)[:, None]
        val_zero = mu[:, None]
        val_pos = (mu + alpha)[:, None]

        d_neg = cw_flat * (w_flat - val_neg) ** 2
        d_zero = cw_flat * (w_flat - val_zero) ** 2
        d_pos = cw_flat * (w_flat - val_pos) ** 2

        dists = np.stack([d_neg, d_zero, d_pos], axis=0)
        assignment = np.argmin(dists, axis=0).astype(np.float64) - 1.0

        cw_sum = cw_flat.sum(axis=1)
        cw_a = (cw_flat * assignment).sum(axis=1)
        cw_a2 = (cw_flat * assignment ** 2).sum(axis=1)
        cw_w = (cw_flat * w_flat).sum(axis=1)
        cw_wa = (cw_flat * w_flat * assignment).sum(axis=1)

        det = cw_sum * cw_a2 - cw_a ** 2
        valid = np.abs(det) > 1e-10

        mu_new = cw_w / np.maximum(cw_sum, 1e-10)
        alpha_new = alpha.copy()

        if valid.any():
            mu_new[valid] = (cw_w[valid] * cw_a2[valid] - cw_wa[valid] * cw_a[valid]) / det[valid]
            alpha_new[valid] = (cw_sum[valid] * cw_wa[valid] - cw_w[valid] * cw_a[valid]) / det[valid]

        all_zero = (cw_a2 < 1e-10)
        if all_zero.any():
            mu_new[all_zero] = cw_w[all_zero] / np.maximum(cw_sum[all_zero], 1e-10)
            alpha_new[all_zero] = 0.0

        mu = mu_new
        alpha = np.maximum(alpha_new, 1e-10)

    w_q = mu[:, None] + alpha[:, None] * assignment
    return w_q.astype(np.float32).reshape(O, I)


def quantize_model_itf_sequential(model, group_size=128, n_iter=20, calibration_data=None, skip_first_k=0):
    """GPTQ-style sequential ITF+AGA quantization.
    
    Args:
        skip_first_k: Keep first K transformer blocks in FP16 (for ablation).
    """
    if calibration_data is None:
        raise ValueError("ITF sequential requires calibration data")

    print(f"  ITF+AGA Sequential: gs={group_size}, iter={n_iter}, cal={len(calibration_data)} samples")
    t0 = time.time()
    total_mse, total_elements, layer_count = 0.0, 0, 0

    # Identify transformer blocks
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        blocks = model.model.layers
        embed = model.model.embed_tokens
        norm = model.model.norm
        lm_head = model.lm_head
    else:
        raise ValueError("Unsupported model architecture")

    n_blocks = len(blocks)
    print(f"  Found {n_blocks} transformer blocks")

    # Get initial hidden states
    hidden_list = []
    with torch.no_grad():
        for batch in calibration_data:
            hidden_list.append(embed(batch).bfloat16())

    # Compute position embeddings
    seq_len = calibration_data[0].shape[1]
    position_ids = torch.arange(seq_len, device="cuda").unsqueeze(0)
    position_embeddings = None
    if hasattr(model.model, 'rotary_emb'):
        with torch.no_grad():
            position_embeddings = model.model.rotary_emb(hidden_list[0], position_ids=position_ids)

    def block_forward(block, h):
        kwargs = {"use_cache": False}
        if position_embeddings is not None:
            kwargs["position_embeddings"] = position_embeddings
            kwargs["position_ids"] = position_ids
        out = block(h, **kwargs)
        result = out[0] if isinstance(out, tuple) else out
        return result.bfloat16()

    # Process each block sequentially
    for block_idx in range(n_blocks):
        block = blocks[block_idx]

        # Skip first K blocks (keep in FP16) for ablation
        if block_idx < skip_first_k:
            new_hidden = []
            with torch.no_grad():
                for h in hidden_list:
                    new_hidden.append(block_forward(block, h))
            hidden_list = new_hidden
            continue

        # Find linear layers in this block
        linears = [(n, m) for n, m in block.named_modules()
                   if isinstance(m, nn.Linear) and m.weight.shape[0] >= 64
                   and m.weight.shape[1] % group_size == 0]

        if not linears:
            new_hidden = []
            with torch.no_grad():
                for h in hidden_list:
                    new_hidden.append(block_forward(block, h))
            hidden_list = new_hidden
            continue

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
                block_forward(block, h)

        for h in hooks:
            h.remove()

        # Quantize layers
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
            layer_count += 1
            module.weight.data = torch.from_numpy(w_q).bfloat16().cuda()

        # Forward through quantized block
        new_hidden = []
        with torch.no_grad():
            for h in hidden_list:
                new_hidden.append(block_forward(block, h))
        hidden_list = new_hidden

        if (block_idx + 1) % 4 == 0 or block_idx == n_blocks - 1:
            avg_mse = total_mse / total_elements if total_elements > 0 else 0
            print(f"  [Block {block_idx+1}/{n_blocks}] layers={layer_count}, avg_mse={avg_mse:.6f} [{time.time()-t0:.0f}s]")

    # Quantize lm_head
    if lm_head.weight.shape[1] % group_size == 0:
        lm_stats = None
        with torch.no_grad():
            for h in hidden_list:
                h_norm = norm(h)
                x = h_norm.reshape(-1, h_norm.shape[-1])
                col_sq = (x.float() ** 2).sum(dim=0).cpu()
                lm_stats = col_sq if lm_stats is None else lm_stats + col_sq

        w = lm_head.weight.data.float().cpu().numpy()
        O, I = w.shape
        col_weights = np.sqrt(lm_stats.numpy() + 1e-10).astype(np.float32)
        w_q = itf_quantize_weight_fast(w, group_size=group_size, n_iter=n_iter, col_weights=col_weights)
        mse = ((w - w_q) ** 2).mean()
        total_mse += mse * O * I
        total_elements += O * I
        layer_count += 1
        lm_head.weight.data = torch.from_numpy(w_q).bfloat16().cuda()

    avg_mse = total_mse / total_elements if total_elements > 0 else 0
    elapsed = time.time() - t0
    print(f"  ITF+AGA done: {layer_count} layers, avg_mse={avg_mse:.6f}, time={elapsed:.0f}s")

    del hidden_list
    torch.cuda.empty_cache()
    return model, avg_mse


# ══════════════════════════════════════════════════════════════════════════════
#  Evaluation
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_ppl(model, tokenizer, num_sequences=80, seq_len=512):
    """WikiText-2 test PPL."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "".join([t for t in ds["text"] if t])
    tokens = tokenizer(text, return_tensors="pt").input_ids[0]

    model.eval()
    total_nll, total_tokens = 0.0, 0
    for i in range(num_sequences):
        start = i * seq_len
        if start + seq_len > len(tokens):
            break
        ids = tokens[start:start + seq_len].unsqueeze(0).cuda()
        out = model(ids, labels=ids)
        total_nll += out.loss.item() * (seq_len - 1)
        total_tokens += seq_len - 1
    ppl = float(torch.exp(torch.tensor(total_nll / total_tokens)).item())
    return ppl


def make_calibration_data(tokenizer, n_samples=128, seq_len=256):
    """WikiText-2 train calibration data."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "".join([t for t in ds["text"] if t])
    tokens = tokenizer(text, return_tensors="pt").input_ids[0]

    batches = []
    for i in range(n_samples):
        start = i * seq_len
        if start + seq_len > len(tokens):
            break
        batches.append(tokens[start:start + seq_len].unsqueeze(0).cuda())
    print(f"  Calibration: {len(batches)} samples, seq_len={seq_len}")
    return batches


# ══════════════════════════════════════════════════════════════════════════════
#  Per-Layer Activation Analysis
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def collect_block_hidden_states(model, samples):
    """Hook each transformer block output, return per-layer hidden states.

    Returns: list of (n_samples, seq_len, d_model) tensors on CPU (fp16).
    """
    blocks = model.model.layers
    n_blocks = len(blocks)

    hidden_states = {i: [] for i in range(n_blocks)}
    hooks = []

    def make_hook(idx):
        def hook_fn(_module, _input, output):
            h = output[0] if isinstance(output, tuple) else output
            hidden_states[idx].append(h.cpu().half())
        return hook_fn

    for i, block in enumerate(blocks):
        hooks.append(block.register_forward_hook(make_hook(i)))

    for sample in samples:
        model(sample)

    for h in hooks:
        h.remove()

    return [torch.stack(hidden_states[i]) for i in range(n_blocks)]


def compute_layer_similarity(fp16_states, ternary_states):
    """Compare FP16 vs ternary hidden states: cosine similarity + MSE per layer."""
    n_layers = len(fp16_states)
    layer_metrics = []

    for i in range(n_layers):
        fp = fp16_states[i].float().flatten(0, 1)   # (N, d_model)
        tq = ternary_states[i].float().flatten(0, 1)

        cos = F.cosine_similarity(fp, tq, dim=-1)
        mse = (fp - tq).pow(2).mean().item()
        fp_var = fp.var(dim=-1).mean().item()
        rel_mse = mse / max(fp_var, 1e-10)

        layer_metrics.append({
            "layer": i,
            "cosine_mean": round(cos.mean().item(), 6),
            "cosine_median": round(cos.median().item(), 6),
            "cosine_min": round(cos.min().item(), 6),
            "mse": round(mse, 6),
            "rel_mse": round(rel_mse, 4),
        })

    return layer_metrics


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="AATQ 7B Full Experiment Suite")
    p.add_argument("--model", default="qwen2-7b", choices=list(MODELS))
    p.add_argument("--gs", type=int, default=128)
    p.add_argument("--ste-steps", type=int, default=500)
    p.add_argument("--ste-lr", type=float, default=0.01)
    p.add_argument("--itf-iter", type=int, default=20)
    p.add_argument("--cal-samples", type=int, default=128)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--output-dir", default="./results_7b")
    p.add_argument("--quick", action="store_true", help="Quick test (fewer steps)")
    p.add_argument("--skip-ste", action="store_true", help="Skip STE experiment")
    p.add_argument("--skip-itf", action="store_true", help="Skip ITF experiment")
    p.add_argument("--analyze-layers", action="store_true", help="Per-layer FP16 vs ternary activation analysis")
    p.add_argument("--analyze-samples", type=int, default=4, help="Num samples for layer analysis (default 4)")
    p.add_argument("--skip-first-k", type=int, default=0, help="Keep first K blocks in FP16 (ablation)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.ste_steps = 200
        args.cal_samples = 64

    model_id = MODELS[args.model]
    os.makedirs(args.output_dir, exist_ok=True)

    tz = timezone(timedelta(hours=8))
    start_time = datetime.now(tz).isoformat()

    print("=" * 70)
    print(f"  AATQ 7B Full Experiment Suite")
    print(f"  Model:  {model_id}")
    print(f"  GPU:    {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB)")
    print(f"  Config: gs={args.gs}, STE steps={args.ste_steps}, ITF iter={args.itf_iter}")
    if args.skip_first_k > 0:
        print(f"  ABLATION: Keeping first {args.skip_first_k} blocks in FP16")
    print(f"  Time:   {start_time}")
    print("=" * 70)

    results = {"model": model_id, "group_size": args.gs, "start_time": start_time,
               "gpu": torch.cuda.get_device_name(0),
               "skip_first_k": args.skip_first_k}

    # ── Load model ───────────────────────────────────────────────────────
    print("\n[Step 1] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.eval()
    print(f"  GPU memory: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # ── FP16 Baseline ────────────────────────────────────────────────────
    print("\n[Step 2] FP16 Baseline PPL...")
    fp16_ppl = evaluate_ppl(model, tokenizer)
    print(f"  FP16 PPL = {fp16_ppl:.2f}")
    results["fp16_ppl"] = round(fp16_ppl, 2)

    # ── Calibration Data ─────────────────────────────────────────────────
    print("\n[Step 3] Preparing calibration data...")
    cal_data = make_calibration_data(tokenizer, n_samples=args.cal_samples, seq_len=args.seq_len)

    # ══════════════════════════════════════════════════════════════════════
    # Experiment A: ITF + AGA (Sequential)
    # ══════════════════════════════════════════════════════════════════════
    if not args.skip_itf:
        print("\n" + "=" * 70)
        print("  EXPERIMENT A: ITF + AGA (Sequential, PT2-LLM style)")
        print("=" * 70)

        # Reload fresh model for ITF
        del model
        torch.cuda.empty_cache()
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
        model.eval()

        # Per-layer: collect FP16 hidden states BEFORE quantization
        fp16_layer_states = None
        analyze_samples = None
        if args.analyze_layers:
            print(f"\n  [Layer Analysis] Collecting FP16 hidden states ({args.analyze_samples} samples)...")
            analyze_samples = cal_data[:args.analyze_samples]
            fp16_layer_states = collect_block_hidden_states(model, analyze_samples)
            print(f"  FP16 states: {len(fp16_layer_states)} layers, shape={fp16_layer_states[0].shape}")

        t_itf = time.time()
        model, avg_mse = quantize_model_itf_sequential(
            model, group_size=args.gs, n_iter=args.itf_iter,
            calibration_data=cal_data, skip_first_k=args.skip_first_k)
        itf_time = time.time() - t_itf

        print("\n  Evaluating ITF+AGA PPL...")
        itf_ppl = evaluate_ppl(model, tokenizer)
        print(f"  ITF+AGA PPL = {itf_ppl:.2f} (time={itf_time:.0f}s)")

        # Per-layer: compare FP16 vs ternary hidden states
        if args.analyze_layers and fp16_layer_states is not None:
            print(f"\n  [Layer Analysis] Collecting ternary hidden states...")
            ternary_layer_states = collect_block_hidden_states(model, analyze_samples)
            print(f"  Computing per-layer similarity...")
            layer_metrics = compute_layer_similarity(fp16_layer_states, ternary_layer_states)
            del fp16_layer_states, ternary_layer_states
            torch.cuda.empty_cache()

            cos_all = [m["cosine_mean"] for m in layer_metrics]
            worst = min(range(len(cos_all)), key=lambda i: cos_all[i])
            print(f"  Cosine similarity: min={min(cos_all):.4f} (L{worst}), "
                  f"median={sorted(cos_all)[len(cos_all)//2]:.4f}, max={max(cos_all):.4f}")

        results["itf_aga"] = {
            "ppl": round(itf_ppl, 2),
            "avg_mse": round(float(avg_mse), 8),
            "time_s": round(itf_time, 1),
        }
        if args.analyze_layers:
            results["itf_aga"]["layer_analysis"] = layer_metrics

    # ══════════════════════════════════════════════════════════════════════
    # Experiment B: STE (Group-wise)
    # ══════════════════════════════════════════════════════════════════════
    if not args.skip_ste:
        print("\n" + "=" * 70)
        print("  EXPERIMENT B: STE (Group-wise, end-to-end)")
        print("=" * 70)

        # Reload fresh model for STE
        del model
        torch.cuda.empty_cache()
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
        model.eval()

        model, n_params = apply_ste_quantization(model, group_size=args.gs)

        t_ste = time.time()
        model, best_loss = train_ste(model, cal_data, steps=args.ste_steps, lr=args.ste_lr)
        ste_time = time.time() - t_ste

        # Freeze and evaluate
        model = freeze_ste(model)
        print("\n  Evaluating STE PPL...")
        ste_ppl = evaluate_ppl(model, tokenizer)
        print(f"  STE PPL = {ste_ppl:.2f} (time={ste_time:.0f}s)")

        results["ste"] = {
            "ppl": round(ste_ppl, 2),
            "best_train_loss": round(best_loss, 4),
            "time_s": round(ste_time, 1),
            "steps": args.ste_steps,
            "lr": args.ste_lr,
            "trainable_params": n_params,
        }

    # ── Save Results ─────────────────────────────────────────────────────
    results["end_time"] = datetime.now(tz).isoformat()

    out_path = os.path.join(args.output_dir, f"{args.model}_gs{args.gs}_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  FINAL RESULTS")
    print("=" * 70)
    print(f"  Model:        {model_id}")
    print(f"  FP16 PPL:     {results['fp16_ppl']}")
    if "itf_aga" in results:
        print(f"  ITF+AGA PPL:  {results['itf_aga']['ppl']} ({results['itf_aga']['time_s']:.0f}s)")
    if "ste" in results:
        print(f"  STE PPL:      {results['ste']['ppl']} ({results['ste']['time_s']:.0f}s)")
    print(f"\n  Results saved to: {out_path}")
    print("  Done!")


if __name__ == "__main__":
    main()
