"""Ternary quantization for transformer models."""

from __future__ import annotations

import torch
import torch.nn as nn
from tqdm import tqdm

from aatq.soft_ternary import (
    SoftTernaryQuantizer,
    compute_initial_delta,
    compute_initial_alpha,
    hard_ternary,
    ste_ternary_projection,
)
from aatq.utils import is_quantizable


def apply_naive_ternary(model: nn.Module, group_size: int | None = None) -> nn.Module:
    """Apply naive ternary quantization to all Linear layers.

    Uses pure statistical initialization: Δ = 0.7·σ, α = mean(|w| | w > Δ).
    No optimization, no calibration data — just a one-shot baseline.

    Args:
        model: HuggingFace causal LM
        group_size: if set, use group-wise quantization

    Returns:
        model with weights replaced by hard ternary values (in-place)
    """
    with torch.no_grad():
        for name, module in tqdm(list(model.named_modules()), desc="Naive ternary quantizing", leave=False):
            if not is_quantizable(module):
                continue
            weight = module.weight.data
            # Skip layers not divisible by group_size
            if group_size is not None and weight.shape[1] % group_size != 0:
                continue
            delta = compute_initial_delta(weight, k=0.7, group_size=group_size)
            alpha = compute_initial_alpha(weight, delta, group_size=group_size)
            w_q = hard_ternary(weight, delta, alpha)

            # Compute zero fraction for logging
            zero_frac = (w_q == 0).float().mean().item()
            tqdm.write(f"  {name}: zero={zero_frac:.1%}, alpha_mean={alpha.mean().item():.4f}")

            module.weight.data = w_q.to(weight.dtype)

    return model


def replace_with_soft_ternary(
    model: nn.Module,
    skip_names: set[str] | None = None,
    use_ste: bool = True,
    group_size: int | None = None,
) -> nn.Module:
    """Replace all Linear layers with SoftTernaryQuantizer wrappers.

    The original Linear layer weights are preserved in the quantizer,
    and the module is replaced so forward passes go through the
    differentiable ternary projection.

    Args:
        model: HuggingFace causal LM
        skip_names: set of child attribute names to skip (e.g. {"lm_head"})
        use_ste: if True (default), use STE mode; if False, use sigmoid mode
        group_size: if set, use group-wise quantization with this group size.
            Each group of `group_size` elements in the input dimension gets
            its own delta and alpha. Set to None for per-channel mode.

    Returns:
        model with Linear layers replaced by SoftTernaryQuantizer (in-place)
    """
    if skip_names is None:
        skip_names = set()

    replacements: dict[str, tuple[nn.Module, nn.Module]] = {}

    for name, parent in model.named_modules():
        for child_name, child in parent.named_children():
            if child_name in skip_names:
                continue
            if is_quantizable(child):
                # Skip layers where in_features is not divisible by group_size
                if group_size is not None and child.in_features % group_size != 0:
                    continue
                full_name = f"{name}.{child_name}" if name else child_name
                replacements[full_name] = (
                    child,
                    SoftTernaryQuantizer(child, use_ste=use_ste, group_size=group_size),
                )

    for name, (original, quantizer) in replacements.items():
        # Navigate to the parent and replace
        if "." in name:
            parent_path, attr = name.rsplit(".", 1)
            parent = model.get_submodule(parent_path)
        else:
            attr = name
            parent = model
        setattr(parent, attr, quantizer)

    return model


def set_all_tau(model: nn.Module, tau: float):
    """Set temperature for all SoftTernaryQuantizer layers."""
    for module in model.modules():
        if isinstance(module, SoftTernaryQuantizer):
            module.set_tau(tau)


def set_all_hard_mode(model: nn.Module, hard: bool):
    """Switch all quantizers between soft/STE and hard mode."""
    for module in model.modules():
        if isinstance(module, SoftTernaryQuantizer):
            module.set_hard_mode(hard)


def set_all_ste_mode(model: nn.Module, ste: bool):
    """Enable/disable STE for all quantizers (only affects training mode)."""
    for module in model.modules():
        if isinstance(module, SoftTernaryQuantizer):
            module.set_ste_mode(ste)


def set_all_mode(model: nn.Module, mode: str):
    """Set mode for all quantizers: 'soft', 'ste', or 'hard'."""
    for module in model.modules():
        if isinstance(module, SoftTernaryQuantizer):
            module.set_mode(mode)


def freeze_to_hard_ternary(model: nn.Module):
    """Convert all SoftTernaryQuantizer layers to frozen hard ternary Linear.

    After optimization is done, this bakes the final hard ternary weights
    into plain nn.Linear modules so the model can be saved/served normally.
    """
    replacements: dict[str, tuple[nn.Module, SoftTernaryQuantizer]] = {}

    for name, parent in model.named_modules():
        for child_name, child in parent.named_children():
            if isinstance(child, SoftTernaryQuantizer):
                full_name = f"{name}.{child_name}" if name else child_name
                replacements[full_name] = (child, child)

    for name, (quantizer, _) in replacements.items():
        w_q = hard_ternary(quantizer.original_weight, quantizer.delta.data, quantizer.alpha.data)
        new_linear = nn.Linear(
            quantizer.in_features,
            quantizer.out_features,
            bias=quantizer.bias is not None,
        )
        new_linear.weight.data = w_q.to(new_linear.weight.dtype)
        if quantizer.bias is not None:
            new_linear.bias.data = quantizer.bias.data

        if "." in name:
            parent_path, attr = name.rsplit(".", 1)
            parent = model.get_submodule(parent_path)
        else:
            attr = name
            parent = model
        setattr(parent, attr, new_linear)

    return model


def collected_quantizer_params(model: nn.Module) -> list[nn.Parameter]:
    """Collect all optimizable Δ and α parameters from quantized layers.

    Returns:
        list of parameters that should be passed to the optimizer
    """
    params = []
    for module in model.modules():
        if isinstance(module, SoftTernaryQuantizer):
            params.append(module.delta)
            params.append(module.alpha)
    return params
