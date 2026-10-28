"""Stage 7 configuration: Flow Matching candidate trajectory generation and risk-aware planning."""
from dataclasses import dataclass
from configs.stage6 import Stage6Config


@dataclass(frozen=True)
class Stage7Config(Stage6Config):
    # Training seeds for reproducibility and paired bootstrap evaluation
    seed: int = 20261021
    training_seeds: tuple = (20261021, 20261022)

    # Base frozen checkpoints
    stage4_checkpoint: str = 'artifacts/restart_stage4/joint_001/best.pt'
    stage6_checkpoint: str = 'artifacts/restart_stage6/runs_orinx/conditioned_20261019/best.pt'

    # Model architecture and parameter budget: Single Orin-X (50M–100M total, target ~60M–80M)
    flow_model_type: str = 'transformer_kinematic'
    flow_d_model: int = 384
    flow_num_layers: int = 5
    flow_num_heads: int = 6
    flow_dim_feedforward: int = 1536
    flow_hidden_dim: int = 256
    flow_euler_steps: int = 10
    flow_noise_dim: int = 40  # 10 waypoints * 4 channels (x, y, v, heading)

    # Risk head architecture (~4M parameters)
    risk_hidden_dim: int = 576
    risk_num_layers: int = 3
    risk_dropout: float = 0.1

    # Candidate planning and scoring
    num_flow_candidates: int = 8
    candidate_sample_seed: int = 42
    cost_weight_progress: float = 1.0
    cost_weight_speed: float = 0.5
    cost_weight_lateral_accel: float = 0.2
    cost_weight_curvature: float = 0.1
    cost_weight_static_bev: float = 5.0
    cost_weight_dynamic_actors: float = 5.0
    cost_weight_world_crash_risk: float = 10.0
    cost_weight_world_offroad_risk: float = 10.0

    # Risk decision threshold: selected by max Youden's J on validation set (ties broken by lower threshold)
    crash_risk_threshold: float = 0.5
    offroad_risk_threshold: float = 0.5

    # Prefix steps for anchor evaluation and branch rollouts (anchor is at exact prefix step)
    prefix_steps: tuple = (20, 50, 80, 110)

    # Roadside prefix context configuration for odd seed indices (last 10 steps before anchor)
    roadside_prefix_offset_m: float = -12.0
    roadside_prefix_speed_ms: float = 12.0
    roadside_prefix_shift_s: float = 0.8
    roadside_prefix_steps: int = 10

    # Training hyper-parameters for Flow Matching & Risk heads
    flow_learning_rate: float = 2e-4
    flow_epochs: int = 25
    risk_learning_rate: float = 1e-3
    risk_epochs: int = 50
    weight_decay: float = 1e-4
    batch_size: int = 64
    mixed_precision: str = 'bf16'
    gradient_clip: float = 5.0

    # Validation and scientific gate criteria
    min_auroc: float = 0.75
    min_test_positive_per_event: int = 20
    min_test_negative_per_event: int = 20
    min_test_seeds: int = 4
    bootstrap_samples: int = 2000

    # Closed-loop evaluation setup: 48 episodes (8 test map-seeds x 3 arms x 2 training seeds)
    closed_loop_seeds: tuple = (1010000, 1020000)
    closed_loop_seeds_per_map: int = 4
    closed_loop_max_steps: int = 1000
    closed_loop_dt_s: float = 0.1
    traffic_density: float = 0.3

    # Artifact and log paths
    artifact_dir: str = 'artifacts/restart_stage7'
    preflight_dir: str = 'artifacts/restart_stage7/preflight_001'


# Disjoint split definition for Stage 7 risk data collection, validation, and testing
STAGE7_SPLITS = {
    'train': dict(maps=('S', 'C', 'SC', 'CS'), seed_start=710000, seeds_per_map=8),
    'val': dict(maps=('CC', 'SS'), seed_start=810000, seeds_per_map=4),
    'test': dict(maps=('SCS', 'CSC'), seed_start=910000, seeds_per_map=4),
}

# 13 Candidate commands spanning safe driving, high speed, and lateral offsets relative to current road position.
# Each entry is (candidate_id, lateral_offset_m, target_speed_m_s, shift_time_s).
# All candidates are strictly 10x4 trajectories executed by the unchanged tracker without hidden controls.
STAGE7_RISK_CANDIDATE_SPECS = (
    ('keep', 0.0, 7.0, 2.0),
    ('slow', 0.0, 3.0, 2.0),
    ('fast', 0.0, 12.0, 2.0),
    ('left_6m', 6.0, 7.0, 1.2),
    ('right_6m', -6.0, 7.0, 1.2),
    ('left_12m', 12.0, 7.0, 1.0),
    ('right_12m', -12.0, 7.0, 1.0),
    ('left_6m_fast', 6.0, 12.0, 1.0),
    ('right_6m_fast', -6.0, 12.0, 1.0),
    ('left_12m_fast', 12.0, 12.0, 0.8),
    ('right_12m_fast', -12.0, 12.0, 0.8),
    ('left_6m_slow', 6.0, 3.0, 1.5),
    ('right_6m_slow', -6.0, 3.0, 1.5),
)
