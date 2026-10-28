"""Frozen stage 3 settings; no action head or later-stage world rollout."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Stage3Config:
    seed: int = 20261017
    epochs: int = 12
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    workers: int = 2
    cpu_threads: int = 6
    mixed_precision: str = 'bf16'
    encoder_dim: int = 256
    encoder_layers: int = 6
    encoder_heads: int = 4
    encoder_mlp: int = 1280
    patch_stride: int = 4  # 32x32 stem -> 8x8 tokens, preserving 128px input
    predictor_dim: int = 384
    predictor_layers: int = 6
    predictor_heads: int = 6
    predictor_mlp: int = 1536
    lambda_sim: float = 25.
    lambda_var: float = 25.
    lambda_cov: float = 1.
    variance_gamma: float = 1.
    variance_epsilon: float = 1e-4
    gradient_clip: float = 5.
    validation_pairs: int = 4096
    bootstrap_samples: int = 2000
    rank_threshold: float = .75
    stable_epochs: int = 3
    minimum_prediction_gain: float = .05
    visualization_interval_steps: int = 400
    visualization_sample_ids: tuple = (0, 512, 2048, 4096)
    visualization_seed: int = 20261017
    sdpa_backend: str = 'AUTO'  # AUTO = PyTorch ROCm selection (AOTriton flash on gfx1100); MATH = pure math for reproducibility
    residual_target: bool = False  # predict future_latent - current_latent (removes copy-current shortcut)
    residual_horizons: tuple = ()  # empty = all; e.g. (0,1) = hybrid residual for +0.5s/+1.0s only
    dynamic_loss_weight: float = 0.0  # upweight BEV tokens where the future differs from the current frame
    masked_completion_weight: float = 0.0
    mask_ratio: float = 0.25
    masked_horizons: tuple = ()  # empty = all four future horizons

