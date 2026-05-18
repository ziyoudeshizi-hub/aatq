"""Utility functions for AATQ."""

from typing import TypeAlias

import torch
import torch.nn as nn

# Linear layers eligible for ternary quantization
QUANTIZABLE_LAYERS: TypeAlias = set[type]
QUANTIZABLE: set[type] = {nn.Linear}


def is_quantizable(module: nn.Module) -> bool:
    """Check if a module is eligible for ternary quantization."""
    return type(module) in QUANTIZABLE


def count_quantizable_layers(model: nn.Module) -> int:
    """Count the number of quantizable linear layers in a model."""
    return sum(1 for m in model.modules() if is_quantizable(m))


def count_parameters(module: nn.Module) -> tuple[int, int]:
    """Count trainable and total parameters.

    Returns:
        (trainable_params, total_params)
    """
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    total = sum(p.numel() for p in module.parameters())
    return trainable, total


def model_size_mb(model: nn.Module, dtype_bytes: int = 2) -> float:
    """Estimate model size in MB.

    Args:
        model: the model
        dtype_bytes: bytes per parameter (2 for fp16/bf16, 4 for fp32)
    """
    return sum(p.numel() for p in model.parameters()) * dtype_bytes / (1024 * 1024)


def ternary_size_mb(model: nn.Module, packed: bool = False) -> dict[str, float]:
    """Estimate ternary model size.

    Arguments:
        model: the model
        packed: if True, estimate 5-trits-per-byte packed size; if False, 2-bit naive

    Returns:
        dict with keys: "weights_mb", "scales_mb", "total_mb", "compression_ratio"
    """
    total_params = sum(p.numel() for p in model.parameters())
    num_channels = sum(m.out_features for m in model.modules() if isinstance(m, nn.Linear))

    bits_per_weight = 8 / 5 if packed else 2  # 5-trits-per-byte or naive 2-bit
    weights_mb = total_params * bits_per_weight / 8 / (1024 * 1024)
    scales_mb = num_channels * 4 / (1024 * 1024)  # fp32 per-channel alpha + delta
    total_mb = weights_mb + scales_mb
    fp16_mb = total_params * 2 / (1024 * 1024)
    compression_ratio = fp16_mb / total_mb if total_mb > 0 else float("inf")

    return {
        "weights_mb": weights_mb,
        "scales_mb": scales_mb,
        "total_mb": total_mb,
        "fp16_mb": fp16_mb,
        "compression_ratio": compression_ratio,
    }
