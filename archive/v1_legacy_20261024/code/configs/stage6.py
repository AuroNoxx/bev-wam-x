"""Stage 6: controlled action-conditioned world prediction, no candidate planner."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Stage6Config:
    seed: int = 20261019
    training_seeds: tuple = (20261019, 20261020)
    checkpoint: str = 'artifacts/restart_stage4/joint_001/best.pt'
    # Four actual future observations now span the action's full 2s horizon.
    future_times_s: tuple = (.5, 1., 1.5, 2.)
    history_dt_s: float = .1
    control_dt_s: float = .1
    force_destroy_between_resets: bool = True
    prefix_steps: tuple = (20, 50, 80, 110)
    candidates: tuple = ('keep', 'left', 'right', 'slow')
    train_seeds_per_map: int = 8
    val_seeds_per_map: int = 4
    test_seeds_per_map: int = 4
    epochs: int = 40
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    cpu_threads: int = 4
    mixed_precision: str = 'bf16'
    gradient_clip: float = 5.
    bootstrap_samples: int = 2000
    minimum_latent_gain: float = .05
    minimum_delta_gain: float = .05
    minimum_retrieval_accuracy: float = .5
    visualization_interval_epochs: int = 5
    replay_state_atol: float = 1e-5
    replay_scene_position_m: float = 1e-5
    replay_scene_heading_rad: float = 1e-5
    replay_scene_speed_m_s: float = 1e-5
    replay_scene_velocity_m_s: float = 1e-5
    # Native fresh-object reset passed all eight long-prefix probe contexts exactly.
    # Restore strict raw image identity; historical relaxed probes remain preserved.
    replay_bev_different_fraction: float = 0.
    # Single Orin-X JEPA-World-X (~75.5M parameters, Scheme C)
    world_model_dim: int = 768
    world_num_layers: int = 8
    world_num_heads: int = 12
    world_hidden_dim: int = 2816
    world_in_dim: int = 256
    world_num_tokens: int = 64
    world_num_future: int = 4


BRANCH_SPLITS = {
    'train': dict(maps=('S', 'C', 'SC', 'CS'), seed_start=410000),
    'val': dict(maps=('CC', 'SS'), seed_start=510000),
    'test': dict(maps=('SCS', 'CSC'), seed_start=610000),
}
