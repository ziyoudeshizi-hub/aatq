#!/usr/bin/env python
"""Block-wise STE for 7B: trains one transformer block at a time to fit in 80GB."""
from __future__ import annotations
import argparse, json, math, os, time
from datetime import datetime, timezone, timedelta

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class STQ_Block(nn.Module):
    """Soft Ternary Quantizer for a single linear layer."""
    def __init__(self, linear, group_size=128):
        super().__init__()
        w = linear.weight.data.float()
        O, I = w.shape
        self.gs = group_size
        self.G = I // group_size
        self.O, self.I = O, I
        self.bias_param = nn.Parameter(linear.bias.data.clone()) if linear.bias is not None else None
        # Store original weight in bf16 to save memory
        self.register_buffer("orig_w", w.to(torch.bfloat16))
        # Initialize delta and alpha
        w_3d = w.view(O, self.G, group_size)
        d = 0.7 * w_3d.std(dim=2)
        self.delta = nn.Parameter(d)
        above = w_3d.abs() > d.unsqueeze(-1)
        a = (w_3d.abs() * above.float()).sum(dim=2) / above.float().sum(dim=2).clamp(min=1)
        self.alpha = nn.Parameter(a)
        self.tau = 1.0

    def forward(self, x):
        w_3d = self.orig_w.float().view(self.O, self.G, self.gs)
        delta_3d = self.delta.unsqueeze(-1)
        alpha_3d = self.alpha.unsqueeze(-1)
        abs_w = w_3d.abs()
        sgn_w = w_3d.sign()
        mask_hard = (abs_w > delta_3d).float()
        sig_in = ((abs_w - delta_3d) / max(self.tau, 0.25)).clamp(-10, 10)
        mask_soft = torch.sigmoid(sig_in)
        mask = mask_hard.detach() + mask_soft - mask_soft.detach()
        wq = (alpha_3d * mask * sgn_w).view(self.O, self.I)
        return F.linear(x, wq.to(x.dtype), self.bias_param.to(x.dtype) if self.bias_param is not None else None)

    def get_hard_weight(self):
        w_3d = self.orig_w.float().view(self.O, self.G, self.gs)
        delta_3d = self.delta.detach().unsqueeze(-1)
        alpha_3d = self.alpha.detach().unsqueeze(-1)
        mask = (w_3d.abs() > delta_3d).float()
        return (alpha_3d * mask * w_3d.sign()).view(self.O, self.I).to(torch.bfloat16)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B")
    p.add_argument("--gs", type=int, default=128)
    p.add_argument("--steps-per-block", type=int, default=50)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--cal-samples", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--output-dir", default="./results_7b")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tz = timezone(timedelta(hours=8))
    t0 = time.time()

    print("=" * 60)
    print("  Block-wise STE for 7B")
    print(f"  Model: {args.model}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Steps/block: {args.steps_per_block}, lr={args.lr}, gs={args.gs}")
    print("=" * 60)

    # Load model
    print("\n[1] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.eval()
    print(f"  GPU mem: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    # Calibration data
    print("\n[2] Calibration data...")
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "".join([t for t in ds["text"] if t])
    tokens = tokenizer(text, return_tensors="pt").input_ids[0]
    cal_batches = []
    for i in range(args.cal_samples):
        start = i * args.seq_len
        if start + args.seq_len > len(tokens):
            break
        cal_batches.append(tokens[start:start+args.seq_len].unsqueeze(0).cuda())
    print(f"  {len(cal_batches)} samples, seq_len={args.seq_len}")

    # Identify model structure
    blocks = model.model.layers
    embed = model.model.embed_tokens
    norm = model.model.norm
    lm_head = model.lm_head
    n_blocks = len(blocks)
    print(f"  {n_blocks} transformer blocks")

    # Get initial hidden states
    print("\n[3] Computing initial hidden states...")
    hidden_list = []
    with torch.no_grad():
        for batch in cal_batches:
            hidden_list.append(embed(batch))

    # Position embeddings
    seq_len = args.seq_len
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
        return out[0] if isinstance(out, tuple) else out

    # Block-wise STE training
    print("\n[4] Block-wise STE training...")
    total_layers = 0

    for block_idx in range(n_blocks):
        block = blocks[block_idx]
        torch.cuda.empty_cache()

        # Find quantizable linears in this block
        linears_info = []
        for name, module in block.named_modules():
            if isinstance(module, nn.Linear) and module.weight.shape[0] >= 64:
                if module.weight.shape[1] % args.gs == 0:
                    linears_info.append((name, module))

        if not linears_info:
            with torch.no_grad():
                hidden_list = [block_forward(block, h) for h in hidden_list]
            continue

        # Compute target outputs BEFORE quantizing (original block behavior)
        target_list = []
        with torch.no_grad():
            for h in hidden_list:
                target_list.append(block_forward(block, h).detach())

        # Replace linears with STQ
        stq_modules = []
        for name, module in linears_info:
            parts = name.split(".")
            parent = block
            for part in parts[:-1]:
                parent = getattr(parent, part)
            stq = STQ_Block(module, args.gs)
            setattr(parent, parts[-1], stq)
            stq_modules.append(stq)
            total_layers += 1

        # Collect trainable params for this block
        params = []
        for stq in stq_modules:
            params.extend([stq.delta, stq.alpha])

        opt = torch.optim.Adam(params, lr=args.lr, betas=(0.9, 0.99))

        # Train this block: minimize ||quantized_block(h) - original_block(h)||^2
        for step in range(args.steps_per_block):
            progress = step / max(args.steps_per_block - 1, 1)
            tau = 0.25 + 0.75 * 0.5 * (1 + math.cos(math.pi * progress))
            for stq in stq_modules:
                stq.tau = tau

            batch_idx = step % len(hidden_list)
            h_in = hidden_list[batch_idx].detach()
            target = target_list[batch_idx]

            h_out = block_forward(block, h_in)
            loss = (h_out - target).pow(2).mean()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

        # Freeze this block: replace STQ with hard ternary Linear
        for name, module in linears_info:
            parts = name.split(".")
            parent = block
            for part in parts[:-1]:
                parent = getattr(parent, part)
            # Find the STQ module
            stq = getattr(parent, parts[-1])
            if isinstance(stq, STQ_Block):
                w_q = stq.get_hard_weight()
                new_linear = nn.Linear(stq.I, stq.O, bias=stq.bias_param is not None,
                                       device="cuda", dtype=torch.bfloat16)
                new_linear.weight.data = w_q
                if stq.bias_param is not None:
                    new_linear.bias.data = stq.bias_param.data.to(torch.bfloat16)
                setattr(parent, parts[-1], new_linear)

        # Forward hidden states through quantized block
        with torch.no_grad():
            hidden_list = [block_forward(block, h) for h in hidden_list]

        torch.cuda.empty_cache()

        if (block_idx + 1) % 4 == 0 or block_idx == n_blocks - 1:
            elapsed = time.time() - t0
            print(f"  [Block {block_idx+1}/{n_blocks}] layers={total_layers} [{elapsed:.0f}s]")

    # Evaluate PPL
    print("\n[5] Evaluating PPL...")
    ds_test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text_test = "".join([t for t in ds_test["text"] if t])
    test_tokens = tokenizer(text_test, return_tensors="pt").input_ids[0]

    model.eval()
    total_nll, total_tok = 0.0, 0
    eval_seq_len = 512
    n_eval = 80
    with torch.no_grad():
        for i in range(n_eval):
            start = i * eval_seq_len
            if start + eval_seq_len > len(test_tokens):
                break
            ids = test_tokens[start:start+eval_seq_len].unsqueeze(0).cuda()
            out = model(ids, labels=ids)
            total_nll += out.loss.item() * (eval_seq_len - 1)
            total_tok += eval_seq_len - 1

    ppl = float(torch.exp(torch.tensor(total_nll / total_tok)).item())
    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"  RESULT: Block-wise STE PPL = {ppl:.2f}")
    print(f"  Time: {elapsed:.0f}s, Layers: {total_layers}")
    print(f"{'='*60}")

    # Save
    results = {
        "method": "blockwise_ste",
        "model": args.model,
        "ppl": round(ppl, 2),
        "time_s": round(elapsed, 1),
        "steps_per_block": args.steps_per_block,
        "lr": args.lr,
        "group_size": args.gs,
        "total_layers": total_layers,
        "timestamp": datetime.now(tz).isoformat(),
    }
    model_tag = args.model.replace("/", "_").replace(".", "-")
    out_path = os.path.join(args.output_dir, f"{model_tag}_ste_blockwise.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {out_path}")


if __name__ == "__main__":
    main()
