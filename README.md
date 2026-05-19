# AATQ — Activation-Aware Ternary Quantization

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Post-training ternary quantization with soft ternary projection and temperature annealing.

## Why

PT²-LLM (ICLR 2026) explicitly gave up on optimizing quantization threshold Δ due to overfitting. We solve this with a differentiable soft ternary projection that makes the loss landscape smooth and optimizable.

## Core insight

Hard ternary (step function) → gradient is zero everywhere → Δ can't be learned.
Soft ternary (tanh) → smooth → gradient flows → Δ becomes optimizable.

Temperature annealing: start with large τ (smooth, exploratory), end with small τ (converge to hard ternary).

## Quick start

```bash
pip install -e .
python scripts/run_baseline.py
```

## License

Apache 2.0
