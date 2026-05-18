"""Soft Ternary Projection — the core differentiable innovation for AATQ.

Hard ternary quantization uses a step function: |w| > Δ → ±α, else 0.
This is non-differentiable at |w| = Δ and has zero gradient elsewhere,
making Δ impossible to optimize with gradient descent.

Two differentiable relaxation strategies are provided:

1. **Sigmoid mode** (original): replaces hard threshold with sigmoid.
   Pro: smooth landscape everywhere.
   Con: "soft-hard gap" — training uses continuous sigmoid, eval uses hard
   threshold, causing divergence with longer training.

2. **STE mode** (recommended): forward uses hard threshold (matches eval
   exactly), backward uses sigmoid gradient approximation via the
   Straight-Through Estimator trick:
       mask = hard.detach() + soft - soft.detach()
   Pro: zero soft-hard gap, enables longer training without degradation.
   Con: gradient is biased (but works well in practice).

Critical: we use sigmoid (not tanh) because tanh inverts the sign for
|w| << Δ (tanh(negative) → -1, mutating -α*sign(w) instead of 0).
Sigmoid correctly maps |w| << Δ → 0 and |w| >> Δ → α·sign(w).

Empirical results on TinyLlama-1.1B (500 steps, CAL=128, SEQ=256):
  - Sigmoid mode: PPL 2405 (degrades to 4075 at 500 steps)
  - STE mode:     PPL 1089 (improves with more steps up to sweet spot)

3. **Group-wise mode**: subdivides each weight row into groups of `group_size`
   elements, giving each group its own delta and alpha. This captures
   intra-row magnitude variation that per-channel mode misses.
   - Group-wise STE (gs=128, lr=0.02): PPL 572 on TinyLlama-1.1B (-47.5%)
"""

import math
import torch
import torch.nn as nn


def soft_ternary_projection(
    weight: torch.Tensor,
    delta: torch.Tensor,
    tau: float,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """Differentiable soft ternary projection.

    w_q = alpha * sigmoid((|w| - delta) / tau) * sign(w)

    Supports both per-channel (delta dim=1) and group-wise (delta dim=2) modes.

    Args:
        weight: original weight tensor, shape [out_features, in_features]
        delta: threshold. Shape [out_features] for per-channel,
               or [out_features, num_groups] for group-wise.
        tau: temperature, larger = smoother
        alpha: scaling factor, same shape as delta.

    Returns:
        soft ternary weight, same shape as weight
    """
    if delta.dim() == 2:
        # Group-wise mode: reshape weight to [O, G, gs]
        O, I = weight.shape
        G = delta.shape[1]
        gs = I // G
        w_3d = weight.view(O, G, gs)
        delta_3d = delta.unsqueeze(-1)  # [O, G, 1]
        alpha_3d = alpha.unsqueeze(-1)  # [O, G, 1]
        soft_mask = torch.sigmoid((w_3d.abs() - delta_3d) / tau)
        w_q = (alpha_3d * soft_mask * w_3d.sign()).view(O, I)
        return w_q

    # Per-channel mode
    abs_w = weight.abs()
    sgn_w = weight.sign()
    delta = delta.view(-1, 1) if delta.dim() <= 1 else delta
    alpha = alpha.view(-1, 1) if alpha.dim() <= 1 else alpha
    soft_mask = torch.sigmoid((abs_w - delta) / tau)
    w_q = alpha * soft_mask * sgn_w
    return w_q


def hard_ternary(
    weight: torch.Tensor,
    delta: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """Hard ternary quantization { -alpha, 0, +alpha } via step function.

    Supports per-channel (delta dim=1) and group-wise (delta dim=2).

    Args:
        weight: original weight tensor
        delta: threshold (per-channel or group-wise)
        alpha: scaling factor (same shape as delta)

    Returns:
        quantized ternary weight
    """
    if delta.dim() == 2:
        O, I = weight.shape
        G = delta.shape[1]
        gs = I // G
        w_3d = weight.view(O, G, gs)
        delta_3d = delta.unsqueeze(-1)
        alpha_3d = alpha.unsqueeze(-1)
        mask = (w_3d.abs() > delta_3d).float()
        return (alpha_3d * w_3d.sign() * mask).view(O, I)

    mask = weight.abs() > delta.view(-1, 1)
    return alpha.view(-1, 1) * weight.sign() * mask.float()


def ste_ternary_projection(
    weight: torch.Tensor,
    delta: torch.Tensor,
    tau: float,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """STE (Straight-Through Estimator) ternary projection.

    Forward uses hard threshold, backward uses sigmoid gradient approximation.
    Supports per-channel (delta dim=1) and group-wise (delta dim=2).

    Args:
        weight: original weight tensor, shape [out_features, in_features]
        delta: threshold. [out_features] for per-channel, [out_features, G] for group-wise.
        tau: temperature for gradient approximation (does NOT affect forward)
        alpha: scaling factor, same shape as delta.

    Returns:
        STE ternary weight, same shape as weight
    """
    if delta.dim() == 2:
        # Group-wise mode
        O, I = weight.shape
        G = delta.shape[1]
        gs = I // G
        w_3d = weight.view(O, G, gs)
        abs_w = w_3d.abs()
        sgn_w = w_3d.sign()
        delta_3d = delta.unsqueeze(-1)  # [O, G, 1]
        alpha_3d = alpha.unsqueeze(-1)  # [O, G, 1]
        mask_hard = (abs_w > delta_3d).float()
        mask_soft = torch.sigmoid((abs_w - delta_3d) / tau)
        mask = mask_hard.detach() + mask_soft - mask_soft.detach()
        w_q = (alpha_3d * mask * sgn_w).view(O, I)
        return w_q

    # Per-channel mode
    abs_w = weight.abs()
    sgn_w = weight.sign()
    delta = delta.view(-1, 1) if delta.dim() <= 1 else delta
    alpha = alpha.view(-1, 1) if alpha.dim() <= 1 else alpha
    mask_hard = (abs_w > delta).float()
    mask_soft = torch.sigmoid((abs_w - delta) / tau)
    mask = mask_hard.detach() + mask_soft - mask_soft.detach()
    w_q = alpha * mask * sgn_w
    return w_q


def compute_initial_delta(weight: torch.Tensor, k: float = 0.7, group_size: int = None) -> torch.Tensor:
    """Initialize delta from weight statistics.

    Per-channel: delta = k * std(row)  → shape [out_features]
    Group-wise:  delta = k * std(group) → shape [out_features, num_groups]

    Args:
        weight: weight tensor [out_features, in_features]
        k: multiplier on stddev
        group_size: if set, compute per-group delta

    Returns:
        delta tensor
    """
    if group_size is not None:
        O, I = weight.shape
        assert I % group_size == 0, f"in_features={I} not divisible by group_size={group_size}"
        G = I // group_size
        w_3d = weight.view(O, G, group_size)
        return k * w_3d.std(dim=2)  # [O, G]
    return k * weight.std(dim=1)


def compute_initial_alpha(weight: torch.Tensor, delta: torch.Tensor, group_size: int = None) -> torch.Tensor:
    """Initialize alpha as the mean of |w| for weights above threshold.

    Supports both per-channel (delta dim=1) and group-wise (delta dim=2).

    Args:
        weight: weight tensor [out_features, in_features]
        delta: threshold (per-channel or group-wise)
        group_size: if set, compute per-group alpha

    Returns:
        alpha tensor (same shape as delta)
    """
    if group_size is not None or delta.dim() == 2:
        O, I = weight.shape
        G = delta.shape[1] if delta.dim() == 2 else I // group_size
        gs = I // G
        w_3d = weight.view(O, G, gs)
        delta_3d = delta.view(O, G, 1) if delta.dim() == 2 else delta.unsqueeze(-1)
        above = w_3d.abs() > delta_3d
        alpha = (w_3d.abs() * above.float()).sum(dim=2) / above.float().sum(dim=2).clamp(min=1)
        return alpha  # [O, G]
    delta_2d = delta.view(-1, 1)
    above = weight.abs() > delta_2d
    alpha = (weight.abs() * above.float()).sum(dim=1) / above.float().sum(dim=1).clamp(min=1)
    return alpha


def temperature_schedule(
    step: int,
    total_steps: int,
    tau_start: float = 1.0,
    tau_end: float = 0.01,
) -> float:
    """Cosine annealing schedule for temperature τ.

    High τ early → smooth landscape, good gradient signal, less overfitting.
    Low τ late → close to hard ternary, ready for inference.

    Args:
        step: current step (0-indexed)
        total_steps: total optimization steps
        tau_start: initial temperature
        tau_end: final temperature

    Returns:
        current tau
    """
    progress = step / max(total_steps - 1, 1)
    tau = tau_end + 0.5 * (tau_start - tau_end) * (1 + math.cos(math.pi * progress))
    return tau


class SoftTernaryQuantizer(nn.Module):
    """Quantizer that wraps a Linear layer with differentiable ternary projection.

    Supports three modes:
      - "soft": pure sigmoid relaxation (original method, for ablation)
      - "ste": Straight-Through Estimator (recommended for training)
      - "hard": hard threshold for inference/evaluation

    Supports per-channel (group_size=None) and group-wise quantization.

    Usage:
        quantizer = SoftTernaryQuantizer(linear, group_size=128)
        quantizer.set_mode("ste")   # for training
        quantizer.set_mode("hard")  # for evaluation
    """

    def __init__(self, linear: nn.Linear, use_ste: bool = True, group_size: int = None):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.group_size = group_size

        if group_size is not None:
            assert linear.in_features % group_size == 0, (
                f"in_features={linear.in_features} not divisible by group_size={group_size}"
            )
            self.num_groups = linear.in_features // group_size
        else:
            self.num_groups = None

        weight = linear.weight.data
        self.register_buffer("original_weight", weight.clone())
        linear.weight.data = linear.weight.data.cpu()
        if linear.bias is not None:
            linear.bias.data = linear.bias.data.cpu()

        delta_init = compute_initial_delta(weight, group_size=group_size)
        self.delta = nn.Parameter(delta_init)

        alpha_init = compute_initial_alpha(weight, delta_init.data, group_size=group_size)
        self.alpha = nn.Parameter(alpha_init)

        self.tau = 1.0
        self.use_ste = use_ste
        self.in_hard_mode = False

        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.data.clone())
        else:
            self.register_buffer("bias", None)

    def set_tau(self, tau: float):
        self.tau = tau

    def set_hard_mode(self, hard: bool):
        self.in_hard_mode = hard

    def set_ste_mode(self, ste: bool):
        """Enable/disable STE. When True, training uses hard forward + soft gradient."""
        self.use_ste = ste

    def set_mode(self, mode: str):
        """Set quantizer mode: 'soft', 'ste', or 'hard'."""
        if mode == "hard":
            self.in_hard_mode = True
        elif mode == "ste":
            self.in_hard_mode = False
            self.use_ste = True
        elif mode == "soft":
            self.in_hard_mode = False
            self.use_ste = False
        else:
            raise ValueError(f"Unknown mode '{mode}', expected 'soft', 'ste', or 'hard'")

    def get_quantized_weight(self) -> torch.Tensor:
        if self.in_hard_mode:
            return hard_ternary(self.original_weight, self.delta, self.alpha)
        elif self.use_ste:
            return ste_ternary_projection(
                self.original_weight, self.delta, self.tau, self.alpha
            )
        else:
            return soft_ternary_projection(
                self.original_weight, self.delta, self.tau, self.alpha
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_q = self.get_quantized_weight().to(x.dtype)
        return nn.functional.linear(x, w_q, self.bias)

    def sparsity(self) -> float:
        """Fraction of weights quantized to zero (always uses hard threshold)."""
        w_q = hard_ternary(self.original_weight, self.delta, self.alpha)
        return (w_q == 0).float().mean().item()
