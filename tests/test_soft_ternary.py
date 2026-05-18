"""Tests for soft_ternary module — the core innovation of AATQ."""

import math
import pytest
import torch

from aatq.soft_ternary import (
    compute_initial_alpha,
    compute_initial_delta,
    hard_ternary,
    soft_ternary_projection,
    ste_ternary_projection,
    temperature_schedule,
    SoftTernaryQuantizer,
)


class TestSoftTernaryProjection:
    """Unit tests for the differentiable soft ternary function."""

    def test_output_shape(self):
        """Output shape matches input."""
        w = torch.randn(64, 128)
        delta = torch.tensor([0.3])
        alpha = torch.tensor([1.0])
        result = soft_ternary_projection(w, delta, tau=0.5, alpha=alpha)
        assert result.shape == w.shape

    def test_output_range(self):
        """Soft ternary output is bounded by [-alpha, +alpha]."""
        w = torch.randn(64, 128) * 3
        delta = torch.tensor([1.0])
        alpha = torch.tensor([2.0])
        result = soft_ternary_projection(w, delta, tau=0.5, alpha=alpha)
        assert result.min() >= -2.1  # small epsilon for float error
        assert result.max() <= 2.1

    def test_high_tau_smooth(self):
        """At high tau, output is smooth (not sparse)."""
        w = torch.randn(256, 512)
        delta = torch.full((256,), 0.0)  # all weights past threshold
        alpha = torch.ones(256)
        result = soft_ternary_projection(w, delta, tau=10.0, alpha=alpha)
        # At high tau, tanh with small internal argument, so values spread
        zeros = (result.abs() < 1e-6).sum().item()
        total = result.numel()
        # Should be very few exact zeros at high tau (smooth regime)
        assert zeros / total < 0.01

    def test_low_tau_approaches_hard(self):
        """As tau → 0, soft ternary approaches hard ternary."""
        w = torch.randn(64, 128) * 2
        delta = torch.tensor([0.7])
        alpha = torch.tensor([1.0])

        soft = soft_ternary_projection(w, delta, tau=1e-5, alpha=alpha)
        hard = hard_ternary(w, delta, alpha)

        # At tau=1e-5, sigmoid is effectively a step function
        # Allow small diff at the boundary |w| ≈ Δ
        diff = (soft - hard).abs().mean().item()
        assert diff < 0.05

    def test_gradient_flows(self):
        """Gradient flows through soft ternary (unlike hard ternary)."""
        w = torch.randn(64, 128, requires_grad=True)
        delta = torch.tensor([0.5], requires_grad=True)
        alpha = torch.tensor([1.0], requires_grad=True)

        result = soft_ternary_projection(w, delta, tau=0.5, alpha=alpha)
        loss = result.sum()
        loss.backward()

        assert delta.grad is not None
        assert delta.grad.abs().sum() > 0  # gradient must be non-zero
        assert alpha.grad is not None
        assert alpha.grad.abs().sum() > 0

    def test_per_channel_delta_and_alpha(self):
        """Per-channel Δ and α shape [out_features] broadcast correctly."""
        out_f, in_f = 128, 256
        w = torch.randn(out_f, in_f)
        delta = torch.rand(out_f)
        alpha = torch.rand(out_f)

        result = soft_ternary_projection(w, delta, tau=0.5, alpha=alpha)
        assert result.shape == (out_f, in_f)


class TestHardTernary:
    """Hard ternary quantization tests."""

    def test_only_three_values(self):
        """Output contains only {-α, 0, +α}."""
        w = torch.randn(64, 128)
        alpha = torch.tensor([2.0])
        delta = torch.tensor([1.0])

        result = hard_ternary(w, delta, alpha)
        unique = result.unique()
        assert all(v in [-2.0, 0.0, 2.0] for v in unique.tolist())

    def test_small_weights_zeroed(self):
        """Weights below threshold are quantized to 0."""
        w = torch.zeros(4, 8) + 0.3  # all below threshold
        delta = torch.tensor([1.0])
        alpha = torch.tensor([1.0])

        result = hard_ternary(w, delta, alpha)
        assert (result == 0).all()

    def test_large_weights_ternarized(self):
        """Weights well above threshold become ±alpha."""
        w = torch.tensor([[5.0, -5.0], [0.1, 0.0]])
        delta = torch.tensor([1.0, 1.0])
        alpha = torch.tensor([2.0, 3.0])

        result = hard_ternary(w, delta, alpha)
        expected = torch.tensor([[2.0, -2.0], [0.0, 0.0]])
        assert torch.equal(result, expected)

    def test_no_gradient(self):
        """Hard ternary has zero gradient (sign and > are non-differentiable)."""
        w = torch.randn(64, 128, requires_grad=True)
        delta = torch.tensor([1.0])
        alpha = torch.tensor([1.0])

        result = hard_ternary(w, delta, alpha)
        loss = result.sum()
        loss.backward()

        assert w.grad is not None
        assert w.grad.abs().sum() == 0.0  # all gradients are zero


class TestInitialization:
    """Weight statistics-based initialization."""

    def test_initial_delta_positive(self):
        """Δ must be positive."""
        w = torch.randn(128, 512)
        delta = compute_initial_delta(w, k=0.7)
        assert (delta > 0).all()

    def test_initial_delta_zero_for_constant_weights(self):
        """Δ = 0 for perfectly constant weights."""
        w = torch.ones(128, 512)
        delta = compute_initial_delta(w)
        assert (delta == 0).all()

    def test_initial_alpha(self):
        """Alpha should be positive when weights above threshold exist."""
        w = torch.randn(128, 512) * 2
        delta = compute_initial_delta(w, k=0.7)
        alpha = compute_initial_alpha(w, delta)
        assert (alpha > 0).all()

    def test_initial_alpha_small_delta(self):
        """When Δ is very small, α ≈ mean(|w|)."""
        w = torch.randn(128, 512) * 2
        delta = torch.full((128,), 0.01)
        alpha = compute_initial_alpha(w, delta)
        abs_mean = w.abs().mean(dim=1)
        # With tiny delta, almost all weights are above threshold
        diff = (alpha - abs_mean).abs().mean()
        assert diff < 0.1


class TestTemperatureSchedule:
    """Cosine annealing schedule."""

    def test_start_value(self):
        tau = temperature_schedule(0, 100, tau_start=1.0, tau_end=0.01)
        assert tau == pytest.approx(1.0)

    def test_end_value(self):
        tau = temperature_schedule(99, 100, tau_start=1.0, tau_end=0.01)
        assert tau == pytest.approx(0.01)

    def test_monotonic_decrease(self):
        values = [temperature_schedule(i, 100) for i in range(100)]
        assert all(values[i] >= values[i + 1] for i in range(99))

    def test_range_bounds(self):
        for i in range(100):
            tau = temperature_schedule(i, 100, tau_start=2.0, tau_end=0.1)
            assert 0.1 <= tau <= 2.0


class TestSoftTernaryQuantizer:
    """Integration test for the SoftTernaryQuantizer wrapper."""

    def test_replaces_linear(self):
        linear = torch.nn.Linear(64, 128)
        x = torch.randn(8, 64)

        quantizer = SoftTernaryQuantizer(linear)
        output = quantizer(x)
        assert output.shape == (8, 128)

    def test_original_weight_preserved(self):
        linear = torch.nn.Linear(64, 128)
        w_before = linear.weight.data.clone()
        quantizer = SoftTernaryQuantizer(linear)
        assert torch.equal(quantizer.original_weight, w_before)

    def test_hard_mode_produces_ternary_weights(self):
        linear = torch.nn.Linear(64, 128)
        quantizer = SoftTernaryQuantizer(linear)
        quantizer.set_hard_mode(True)
        w_q = quantizer.get_quantized_weight()
        alpha = quantizer.alpha
        # Each row i should only contain values from {-alpha[i], 0, +alpha[i]}
        for i in range(w_q.shape[0]):
            row_vals = w_q[i].unique()
            expected = {-alpha[i].item(), 0.0, alpha[i].item()}
            assert all(v in expected for v in row_vals.tolist()), (
                f"Row {i}: got {row_vals.tolist()}, expected subset of {expected}"
            )

    def test_sparsity_is_float(self):
        linear = torch.nn.Linear(64, 128)
        quantizer = SoftTernaryQuantizer(linear)
        sparsity = quantizer.sparsity()
        assert 0.0 <= sparsity <= 1.0

    def test_ste_mode_default(self):
        """STE mode is enabled by default."""
        linear = torch.nn.Linear(64, 128)
        quantizer = SoftTernaryQuantizer(linear)
        assert quantizer.use_ste is True

    def test_set_mode_api(self):
        """set_mode() correctly switches between all three modes."""
        linear = torch.nn.Linear(64, 128)
        quantizer = SoftTernaryQuantizer(linear)

        quantizer.set_mode("soft")
        assert not quantizer.in_hard_mode
        assert not quantizer.use_ste

        quantizer.set_mode("ste")
        assert not quantizer.in_hard_mode
        assert quantizer.use_ste

        quantizer.set_mode("hard")
        assert quantizer.in_hard_mode

    def test_set_mode_invalid(self):
        """set_mode() raises on unknown mode."""
        linear = torch.nn.Linear(64, 128)
        quantizer = SoftTernaryQuantizer(linear)
        with pytest.raises(ValueError):
            quantizer.set_mode("unknown")


class TestSTETernaryProjection:
    """Tests for the STE (Straight-Through Estimator) ternary projection."""

    def test_output_shape(self):
        """Output shape matches input."""
        w = torch.randn(64, 128)
        delta = torch.tensor([0.3])
        alpha = torch.tensor([1.0])
        result = ste_ternary_projection(w, delta, tau=0.5, alpha=alpha)
        assert result.shape == w.shape

    def test_forward_matches_hard(self):
        """STE forward output is identical to hard ternary."""
        w = torch.randn(64, 128)
        delta = torch.rand(64) * 0.5 + 0.1
        alpha = torch.rand(64) * 0.5 + 0.5

        ste_result = ste_ternary_projection(w, delta, tau=0.5, alpha=alpha)
        hard_result = hard_ternary(w, delta, alpha)

        # Forward values must be exactly equal (STE uses hard.detach())
        assert torch.allclose(ste_result, hard_result, atol=1e-6)

    def test_gradient_flows_through_delta(self):
        """Gradient flows through delta via STE (unlike hard ternary)."""
        w = torch.randn(64, 128)
        delta = (torch.rand(64) * 0.5 + 0.1).requires_grad_(True)
        alpha = (torch.rand(64) * 0.5 + 0.5).requires_grad_(True)

        result = ste_ternary_projection(w, delta, tau=0.5, alpha=alpha)
        loss = result.sum()
        loss.backward()

        assert delta.grad is not None
        assert delta.grad.abs().sum() > 0
        assert alpha.grad is not None
        assert alpha.grad.abs().sum() > 0

    def test_tau_affects_gradient_not_forward(self):
        """Different tau values produce same forward but different gradients."""
        w = torch.randn(64, 128)
        delta_base = torch.rand(64) * 0.5 + 0.1
        delta_1 = delta_base.clone().requires_grad_(True)
        delta_2 = delta_base.clone().requires_grad_(True)
        alpha = torch.rand(64)

        # Same forward regardless of tau
        out_1 = ste_ternary_projection(w, delta_1, tau=0.1, alpha=alpha)
        out_2 = ste_ternary_projection(w, delta_2, tau=2.0, alpha=alpha)
        assert torch.allclose(out_1, out_2, atol=1e-6)

        # But different gradients
        out_1.sum().backward()
        out_2.sum().backward()
        assert not torch.allclose(delta_1.grad, delta_2.grad)

    def test_per_channel(self):
        """Per-channel delta and alpha work correctly."""
        out_f, in_f = 128, 256
        w = torch.randn(out_f, in_f)
        delta = torch.rand(out_f) * 0.5 + 0.1
        alpha = torch.rand(out_f) * 0.5 + 0.5

        result = ste_ternary_projection(w, delta, tau=0.5, alpha=alpha)
        assert result.shape == (out_f, in_f)

    def test_quantizer_ste_matches_hard_forward(self):
        """SoftTernaryQuantizer in STE mode produces same output as hard mode."""
        linear = torch.nn.Linear(64, 128)
        x = torch.randn(4, 64)

        q = SoftTernaryQuantizer(linear, use_ste=True)
        q.set_tau(0.5)

        # STE forward
        out_ste = q(x)

        # Hard forward
        q.set_mode("hard")
        out_hard = q(x)

        assert torch.allclose(out_ste, out_hard, atol=1e-5)


class TestGroupWise:
    """Tests for group-wise quantization (Phase 2)."""

    def test_compute_initial_delta_group_shape(self):
        """Group-wise delta has shape [O, num_groups]."""
        w = torch.randn(128, 512)
        delta = compute_initial_delta(w, k=0.7, group_size=128)
        assert delta.shape == (128, 4)  # 512 / 128 = 4 groups

    def test_compute_initial_delta_group_positive(self):
        """Group-wise delta must be positive."""
        w = torch.randn(64, 256)
        delta = compute_initial_delta(w, k=0.7, group_size=64)
        assert (delta > 0).all()

    def test_compute_initial_alpha_group_shape(self):
        """Group-wise alpha has shape [O, num_groups]."""
        w = torch.randn(128, 512)
        delta = compute_initial_delta(w, k=0.7, group_size=128)
        alpha = compute_initial_alpha(w, delta, group_size=128)
        assert alpha.shape == (128, 4)
        assert (alpha > 0).all()

    def test_hard_ternary_group_wise(self):
        """Group-wise hard ternary produces valid ternary output."""
        w = torch.randn(64, 256)
        delta = compute_initial_delta(w, k=0.7, group_size=128)
        alpha = compute_initial_alpha(w, delta, group_size=128)
        w_q = hard_ternary(w, delta, alpha)
        assert w_q.shape == w.shape
        # Each group-row should only contain {-alpha_g, 0, +alpha_g}
        O, I = w.shape
        gs = 128
        G = I // gs
        w_3d = w_q.view(O, G, gs)
        for i in range(O):
            for g in range(G):
                vals = w_3d[i, g].unique()
                a = alpha[i, g].item()
                expected = {-a, 0.0, a}
                assert all(
                    any(abs(v - e) < 1e-5 for e in expected) for v in vals.tolist()
                )

    def test_soft_ternary_group_wise_shape(self):
        """Group-wise soft ternary maintains output shape."""
        w = torch.randn(64, 256)
        delta = compute_initial_delta(w, k=0.7, group_size=128)
        alpha = compute_initial_alpha(w, delta, group_size=128)
        result = soft_ternary_projection(w, delta, tau=0.5, alpha=alpha)
        assert result.shape == w.shape

    def test_ste_group_wise_forward_matches_hard(self):
        """Group-wise STE forward == hard ternary forward."""
        w = torch.randn(64, 256)
        delta = compute_initial_delta(w, k=0.7, group_size=128)
        alpha = compute_initial_alpha(w, delta, group_size=128)
        ste_out = ste_ternary_projection(w, delta, tau=0.5, alpha=alpha)
        hard_out = hard_ternary(w, delta, alpha)
        assert torch.allclose(ste_out, hard_out, atol=1e-6)

    def test_ste_group_wise_gradient_flows(self):
        """Gradient flows through group-wise STE delta and alpha."""
        w = torch.randn(64, 256)
        delta = compute_initial_delta(w, k=0.7, group_size=128).requires_grad_(True)
        alpha = compute_initial_alpha(w, delta.detach(), group_size=128).requires_grad_(True)
        result = ste_ternary_projection(w, delta, tau=0.5, alpha=alpha)
        loss = result.sum()
        loss.backward()
        assert delta.grad is not None
        assert delta.grad.abs().sum() > 0
        assert alpha.grad is not None
        assert alpha.grad.abs().sum() > 0

    def test_quantizer_group_wise_init(self):
        """SoftTernaryQuantizer initializes correctly with group_size."""
        linear = torch.nn.Linear(256, 128)
        q = SoftTernaryQuantizer(linear, group_size=128)
        assert q.group_size == 128
        assert q.num_groups == 2  # 256 / 128
        assert q.delta.shape == (128, 2)
        assert q.alpha.shape == (128, 2)

    def test_quantizer_group_wise_forward(self):
        """Group-wise quantizer forward pass works."""
        linear = torch.nn.Linear(256, 128)
        x = torch.randn(4, 256)
        q = SoftTernaryQuantizer(linear, group_size=128)
        output = q(x)
        assert output.shape == (4, 128)

    def test_quantizer_group_wise_hard_mode(self):
        """Group-wise quantizer hard mode produces ternary output."""
        linear = torch.nn.Linear(256, 128)
        x = torch.randn(4, 256)
        q = SoftTernaryQuantizer(linear, group_size=128)
        q.set_mode("hard")
        output = q(x)
        assert output.shape == (4, 128)
        # Verify weight is ternary
        w_q = q.get_quantized_weight()
        # Non-zero values should match group alpha
        O, I = w_q.shape
        w_3d = w_q.view(O, 2, 128)
        for i in range(min(O, 4)):  # spot check first 4 rows
            for g in range(2):
                nonzero = w_3d[i, g][w_3d[i, g] != 0]
                if len(nonzero) > 0:
                    assert torch.allclose(
                        nonzero.abs(), 
                        torch.full_like(nonzero, q.alpha[i, g].item()),
                        atol=1e-5,
                    )

    def test_quantizer_group_wise_sparsity(self):
        """Group-wise quantizer reports valid sparsity."""
        linear = torch.nn.Linear(256, 128)
        q = SoftTernaryQuantizer(linear, group_size=128)
        s = q.sparsity()
        assert 0.0 < s < 1.0

    def test_group_size_assertion(self):
        """group_size must divide in_features evenly."""
        linear = torch.nn.Linear(100, 64)  # 100 not divisible by 128
        with pytest.raises(AssertionError):
            SoftTernaryQuantizer(linear, group_size=128)
