"""Tests for quantize module — high-level API for model quantization."""

import torch
import torch.nn as nn
import pytest

from aatq.quantize import (
    apply_naive_ternary,
    replace_with_soft_ternary,
    set_all_tau,
    set_all_hard_mode,
    set_all_mode,
    freeze_to_hard_ternary,
    collected_quantizer_params,
)
from aatq.soft_ternary import SoftTernaryQuantizer


class TinyModel(nn.Module):
    """Minimal transformer-like model for testing."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(100, 128)
        self.layer1 = nn.Linear(128, 256)
        self.layer2 = nn.Linear(256, 128)
        self.head = nn.Linear(128, 100)

    def forward(self, x):
        x = self.embed(x)
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return self.head(x)


class TestReplaceWithSoftTernary:
    """Tests for replace_with_soft_ternary API."""

    def test_basic_replacement(self):
        model = TinyModel()
        replace_with_soft_ternary(model)
        assert isinstance(model.layer1, SoftTernaryQuantizer)
        assert isinstance(model.layer2, SoftTernaryQuantizer)
        assert isinstance(model.head, SoftTernaryQuantizer)

    def test_skip_names(self):
        model = TinyModel()
        replace_with_soft_ternary(model, skip_names={"head"})
        assert isinstance(model.layer1, SoftTernaryQuantizer)
        assert isinstance(model.layer2, SoftTernaryQuantizer)
        assert isinstance(model.head, nn.Linear)

    def test_group_wise_replacement(self):
        model = TinyModel()
        replace_with_soft_ternary(model, group_size=128)
        # layer1: in=128 (128/128=1 group), should work
        assert isinstance(model.layer1, SoftTernaryQuantizer)
        assert model.layer1.group_size == 128
        assert model.layer1.num_groups == 1
        # layer2: in=256 (256/128=2 groups)
        assert isinstance(model.layer2, SoftTernaryQuantizer)
        assert model.layer2.group_size == 128
        assert model.layer2.num_groups == 2

    def test_group_wise_skips_incompatible(self):
        """Layers with in_features not divisible by group_size are skipped."""
        model = nn.Sequential(
            nn.Linear(100, 64),  # 100 % 128 != 0 → skipped
            nn.Linear(128, 64),  # 128 % 128 == 0 → quantized
        )
        replace_with_soft_ternary(model, group_size=128)
        assert isinstance(model[0], nn.Linear)  # skipped
        assert isinstance(model[1], SoftTernaryQuantizer)  # quantized

    def test_forward_still_works(self):
        model = TinyModel()
        replace_with_soft_ternary(model, group_size=128)
        x = torch.randint(0, 100, (2, 8))
        output = model(x)
        assert output.shape == (2, 8, 100)

    def test_collected_params_group_wise(self):
        model = TinyModel()
        replace_with_soft_ternary(model, group_size=128)
        params = collected_quantizer_params(model)
        # 3 layers * 2 params (delta + alpha) = 6
        assert len(params) == 6
        # Check shapes: layer1 delta should be [256, 1], layer2 delta [128, 2]
        # layer1: out=256, in=128, gs=128 → groups=1 → delta shape [256, 1]
        assert params[0].shape == (256, 1)  # layer1 delta
        assert params[1].shape == (256, 1)  # layer1 alpha
        assert params[2].shape == (128, 2)  # layer2 delta
        assert params[3].shape == (128, 2)  # layer2 alpha

    def test_set_all_mode_group_wise(self):
        model = TinyModel()
        replace_with_soft_ternary(model, group_size=128)
        set_all_mode(model, "hard")
        for m in model.modules():
            if isinstance(m, SoftTernaryQuantizer):
                assert m.in_hard_mode

    def test_freeze_to_hard_group_wise(self):
        model = TinyModel()
        replace_with_soft_ternary(model, group_size=128)
        freeze_to_hard_ternary(model)
        # After freezing, all should be plain Linear again
        assert isinstance(model.layer1, nn.Linear)
        assert isinstance(model.layer2, nn.Linear)
        assert isinstance(model.head, nn.Linear)


class TestApplyNaiveTernary:
    """Tests for apply_naive_ternary with group_size."""

    def test_basic(self):
        model = TinyModel()
        apply_naive_ternary(model)
        # Weights should be ternary
        w = model.layer1.weight.data
        unique_abs = w.abs().unique()
        # Should have 0 and one positive value per row (approximately)
        assert 0.0 in unique_abs.tolist() or (w == 0).any()

    def test_group_wise(self):
        model = TinyModel()
        apply_naive_ternary(model, group_size=128)
        # Weights should still be valid
        w = model.layer2.weight.data
        assert w.shape == (128, 256)
        # Should have zeros
        assert (w == 0).any()
