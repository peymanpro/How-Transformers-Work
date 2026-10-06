# Changelog

## 0.2.0 — 2026-10-06

### Added

- Trainable multi-head self-attention with per-head Q/K/V projections.
- Analytical backward propagation through attention input, Q, K, V, and output projection parameters.
- Finite-difference verification for the trainable attention gradients.
- Training-step coverage that confirms Q/K/V parameters actually update.
- A dedicated trainable multi-head inspection demo.
- Transformer architecture visuals and a README hero animation.
- GitHub Actions quality checks with pytest, Ruff, and mypy.
- MIT license.

### Changed

- The main language-model path now learns attention projections instead of using synthetic fixed routing.
- Training history now reports the loss after each epoch's parameter update.
- README documentation was updated to distinguish the trainable model from the retained synthetic baseline.

### Scope

This release remains a small educational Transformer-style language model. It does not claim GPT-scale training, distributed execution, GPU kernels, KV caching, or the full encoder-decoder architecture with encoder-to-decoder cross-attention.
