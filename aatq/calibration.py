"""Calibration data loading for post-training quantization."""

from typing import Iterator

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


def load_wikitext_samples(
    tokenizer: AutoTokenizer,
    split: str = "train",
    num_samples: int = 512,
    seq_len: int = 2048,
    seed: int = 42,
) -> list[dict]:
    """Load calibration samples from WikiText-2.

    Args:
        tokenizer: HuggingFace tokenizer
        split: dataset split ("train" for calibration, "test" for eval)
        num_samples: number of sequences to return
        seq_len: sequence length (padded/truncated)
        seed: random seed for shuffling

    Returns:
        list of tokenized samples, each with input_ids and attention_mask
    """
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split, trust_remote_code=False)

    # Filter out empty lines and group into full-text
    texts = [entry["text"] for entry in dataset if entry["text"].strip()]

    # Concatenate and chunk into fixed-length sequences
    full_text = tokenizer.eos_token.join(texts)
    tokens = tokenizer(full_text, truncation=False, return_tensors="pt")["input_ids"][0]

    # Chunk into seq_len windows
    samples = []
    rng = torch.Generator().manual_seed(seed)

    for _ in range(num_samples):
        # Random start position
        max_start = max(0, len(tokens) - seq_len)
        if max_start == 0:
            break
        start = torch.randint(0, max_start, (1,), generator=rng).item()
        chunk = tokens[start : start + seq_len]

        if len(chunk) < seq_len:
            chunk = torch.nn.functional.pad(chunk, (0, seq_len - len(chunk)), value=tokenizer.pad_token_id or 0)

        samples.append({
            "input_ids": chunk,
            "attention_mask": torch.ones_like(chunk),
        })

    return samples


def calibration_iterator(
    samples: list[dict],
    batch_size: int = 1,
    device: str = "cuda",
) -> Iterator[dict]:
    """Yield batches of calibration samples on the target device.

    Args:
        samples: list of tokenized samples
        batch_size: batch size
        device: target device

    Yields:
        dict with input_ids and attention_mask on device
    """
    for i in range(0, len(samples), batch_size):
        batch = samples[i : i + batch_size]
        yield {
            "input_ids": torch.stack([s["input_ids"] for s in batch]).to(device),
            "attention_mask": torch.stack([s["attention_mask"] for s in batch]).to(device),
        }
