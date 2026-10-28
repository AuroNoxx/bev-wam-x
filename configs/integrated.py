"""Stage7 integrated v1 contracts; independent of historical stage7 configs."""
from dataclasses import dataclass


@dataclass(frozen=True)
class IntegratedConfig:
    schema_version: int = 3
    navigation_points: int = 8
    navigation_spacing_m: float = 5.0
    preferred_lane: int = 1
    control_dt_s: float = 0.1
    physics_dt_s: float = 0.02
    waypoint_dt_s: float = 0.2
    trajectory_points: int = 10
    target_speed_m_s: float = 7.0
    brake_m_s2: float = 4.0
    max_lateral_accel_m_s2: float = 3.0
    max_static_cost: float = 0.05
    max_dynamic_cost: float = 0.01
    train_seed_start: int = 4200000
    validation_seed_start: int = 4300000
    test_seed_start: int = 4400000
    probe_seed_start: int = 4100000
    encoder_checkpoint: str = 'artifacts/restart_stage4/joint_001/best.pt'
    training_seeds: tuple = (20261023, 20261024)

