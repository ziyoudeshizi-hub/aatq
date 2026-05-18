"""Perplexity evaluation for quantized models."""

from typing import Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.no_grad()
def evaluate_perplexity(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    split: str = "test",
    max_samples: Optional[int] = None,
    stride: int = 512,
    max_length: int = 2048,
    device: str = "cuda",
) -> float:
    """Evaluate perplexity on a text dataset using sliding window.

    Uses the standard HuggingFace perplexity evaluation recipe:
    stride over the text, compute loss on each window, aggregate
    with proper token weighting.

    Args:
        model: HuggingFace causal LM (or quantized version)
        tokenizer: associated tokenizer
        dataset_name: HuggingFace dataset name
        dataset_config: dataset configuration
        split: which split to evaluate on
        max_samples: cap on number of samples (None = all)
        stride: stride for sliding window
        max_length: max sequence length per window
        device: target device

    Returns:
        perplexity (float, lower is better)
    """
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, dataset_config, split=split, trust_remote_code=False)
    texts = [entry["text"] for entry in dataset if entry["text"].strip()]
    if max_samples:
        texts = texts[:max_samples]

    full_text = tokenizer.eos_token.join(texts)
    tokens = tokenizer(full_text, truncation=False, return_tensors="pt")["input_ids"][0]

    model.eval()
    model = model.to(device)

    nlls = []
    prev_end_loc = 0

    pbar = tqdm(range(0, len(tokens), stride), desc="Evaluating perplexity", leave=False)
    for begin_loc in pbar:
        end_loc = min(begin_loc + max_length, len(tokens))
        trg_len = end_loc - prev_end_loc  # may differ from stride on last window

        input_ids = tokens[begin_loc:end_loc].unsqueeze(0).to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100  # only score the new tokens

        outputs = model(input_ids, labels=target_ids)
        neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood.item())
        prev_end_loc = end_loc

        if end_loc >= len(tokens):
            break

    total_nll = sum(nlls)
    total_tokens = len(tokens)
    avg_nll = total_nll / total_tokens
    ppl = float(torch.exp(torch.tensor(avg_nll)).item())

    return ppl


@torch.no_grad()
def evaluate_perplexity_fast(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    num_sequences: int = 50,
    seq_len: int = 2048,
    device: str = "cuda",
) -> float:
    """Quick perplexity estimate using random chunks (much faster).

    Useful for rapid iteration during development — not a replacement
    for the full sliding-window evaluation, but close enough for A/B
    comparisons.

    Args:
        model: causal LM
        tokenizer: tokenizer
        num_sequences: number of random chunks to evaluate
        seq_len: sequence length per chunk
        device: target device

    Returns:
        approximate perplexity
    """
    from aatq.calibration import load_wikitext_samples

    samples = load_wikitext_samples(tokenizer, split="test", num_samples=num_sequences, seq_len=seq_len, seed=0)
    model.eval()
    model = model.to(device)

    total_nll = 0.0
    total_tokens = 0

    for sample in samples:
        input_ids = sample["input_ids"].unsqueeze(0).to(device)
        outputs = model(input_ids, labels=input_ids)
        total_nll += outputs.loss.item() * input_ids.numel()
        total_tokens += input_ids.numel()

    ppl = float(torch.exp(torch.tensor(total_nll / total_tokens)).item())
    return ppl
