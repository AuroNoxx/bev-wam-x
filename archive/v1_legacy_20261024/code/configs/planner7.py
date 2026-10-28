"""Prespecified scoring and rejection rules; frozen before closed-loop evaluation."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Planner7Config:
    target_speed_m_s: float = 7.
    max_speed_m_s: float = 15.
    max_accel_m_s2: float = 8.
    max_heading_error_rad: float = .55
    emergency_deceleration_m_s2: float = 4.
    actor_margin_m: float = .5
    actor_uncertainty_m_per_s: float = .25
    progress_weight: float = .3
    speed_weight: float = 1.
    lateral_accel_weight: float = .5
    curvature_weight: float = .2
    static_weight: float = 5.
    dynamic_weight: float = 5.
    crash_weight: float = 10.
    offroad_weight: float = 10.
