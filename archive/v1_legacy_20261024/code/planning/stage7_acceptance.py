"""Stage 7 Acceptance Framework & Metrics Specification.

Protocol: docs/BEV_WA_ALIGNED_PROTOCOL.md & Updated Multi-Scenario Acceptance Matrix.

Scenario Categories:
1. obstacle_free_navigation (无障碍导航)
   - Verified Ability: 沿路线行驶、终点到达
   - Success Standard: 安全到达 (arrive_dest=True, crash=False, offroad=False), 效率合理 (no timeout, stagnation <= 10)
2. moving_slow_vehicle_no_pass (不允许超车的移动慢车)
   - Verified Ability: 安全跟随、减速与再启动
   - Success Standard: 全程无碰撞、无越界 (crash=False, offroad=False), 保持合理车距 (min_gap >= 2.0m); 允许超时未到达
3. moving_slow_vehicle_pass_allowed (允许绕行的慢车)
   - Verified Ability: 判断许可、绕行、回归
   - Success Standard: 安全超过目标车辆 (overtake_completed=True) 并回归要求的路线车道 (returned_to_route_lane=True), 全程无碰撞、无越界
4. static_full_blockage (静态完全阻塞)
   - Verified Ability: 识别阻塞、停车等待
   - Success Standard: 障碍物前停稳 (stopped_before_obstacle=True, speed < 0.15 m/s), 保持安全距离 (2.0m <= gap <= 15.0m), 全程无碰撞、无越界
5. blockage_clearance_resume (阻塞解除)
   - Verified Ability: 恢复行驶
   - Success Standard: 障碍解除后正常启动 (resumed_after_clearance=True, speed > 2.0 m/s), 无持续死锁 (deadlocked=False)
6. lateral_perturbation_curve_hazard (横向扰动／弯道危险)
   - Verified Ability: 纠偏、控速与恢复
   - Success Standard: 实际扰动达到设定幅度 (actual_perturbation >= 1.0m), 模型参与闭环决策, 满足冻结门槛 (recovered_to_center=True, |lat| < 0.35m within step limit, curve speed controlled without spinout/offroad)
"""

import numpy as np


def categorize_scenario(scene):
    """Categorizes a scenario dictionary into one of the 6 formal acceptance categories."""
    case = scene.get("case")
    is_controlled = "case" in scene
    map_name = scene.get("map", "")
    lanes = scene.get("lanes", 1)
    stationary = scene.get("stationary", False)

    if not is_controlled:
        return "obstacle_free_navigation"

    if case in ("blocked_static", "static_blockage") or (case == "blocked" and stationary):
        return "static_full_blockage"
    elif case in ("blockage_resume", "blockage_clearance_resume"):
        return "blockage_clearance_resume"
    elif case in ("allowed_left", "allowed_right", "pass_allowed"):
        return "moving_slow_vehicle_pass_allowed"
    elif case in ("no_adjacent_lane", "blocked_moving", "blocked"):
        # If lanes == 1 or all lanes blocked by moving vehicles, overtaking is not permitted / not feasible
        return "moving_slow_vehicle_no_pass"
    elif "perturbation" in str(case) or "curve_hazard" in str(case):
        return "lateral_perturbation_curve_hazard"

    return "obstacle_free_navigation"


def evaluate_scenario_outcome(category, ep_summary, trajectory_log=None):
    """Evaluates scenario outcome under the strict per-category success criteria.

    Args:
        category: str in the 6 categories
        ep_summary: dict containing raw simulation flags:
            arrived, crashed, offroad, timeout, steps, route_completion,
            stagnation_steps, overtake_vehicle_num, etc.
        trajectory_log: optional list of step records for detailed physical evaluation.

    Returns:
        dict with success (bool), category, failure_reason (str or None),
        effective_route_completion (float), and category-specific diagnostics.
    """
    arrived = bool(ep_summary.get("arrived", False))
    crashed = bool(ep_summary.get("crashed", False))
    offroad = bool(ep_summary.get("offroad", False))
    timeout = bool(ep_summary.get("timeout", False))
    raw_rc = float(ep_summary.get("route_completion", 0.0))
    stagnation_steps = int(ep_summary.get("stagnation_steps", 0))

    # Basic safety filter: crash or off-road ALWAYS invalidates driving success!
    # Furthermore, route completion cannot reward aggressive progression before crashing.
    effective_rc = raw_rc if (not crashed and not offroad) else 0.0

    diag = {}
    success = False
    failure_reason = None

    if crashed:
        failure_reason = "vehicle_collision"
    elif offroad:
        failure_reason = "road_boundary_violation"

    if category == "obstacle_free_navigation":
        # Standard: Arrive safely without crash, offroad, or timeout; reasonable efficiency
        if not crashed and not offroad:
            if arrived and not timeout and stagnation_steps <= 10:
                success = True
            elif timeout:
                failure_reason = "timeout_without_arrival"
            elif stagnation_steps > 10:
                failure_reason = "excessive_stagnation"
            else:
                failure_reason = "incomplete_navigation"
        diag = {
            "arrived": arrived,
            "stagnation_steps": stagnation_steps,
            "timeout": timeout,
        }

    elif category == "moving_slow_vehicle_no_pass":
        # Standard: Safe car-following; no crash, no offroad, safe gap maintained; TIMEOUT ALLOWED!
        min_gap = ep_summary.get("min_longitudinal_gap_m")
        if min_gap is None and trajectory_log:
            gaps = [r["closest_npc"]["longitudinal_gap_m"] for r in trajectory_log if r.get("closest_npc")]
            min_gap = min(gaps) if gaps else None

        safe_gap_maintained = (min_gap is not None and min_gap >= 2.0) if min_gap is not None else (not crashed)
        if not crashed and not offroad:
            if safe_gap_maintained:
                success = True
            else:
                failure_reason = f"tailgating_too_close_min_gap_{min_gap:.2f}m"
        diag = {
            "min_longitudinal_gap_m": min_gap,
            "safe_gap_maintained": safe_gap_maintained,
            "timeout_permitted": True,
        }

    elif category == "moving_slow_vehicle_pass_allowed":
        # Standard: Safely overtake the lead vehicle and return to target route lane
        overtake_completed = bool(ep_summary.get("overtake_completed", False))
        returned_to_route_lane = bool(ep_summary.get("returned_to_route_lane", False))

        if trajectory_log and (not overtake_completed or not returned_to_route_lane):
            # Inspect step trajectory for overtaking and return
            passed_lead = False
            returned_lane = False
            for r in trajectory_log:
                npc = r.get("closest_npc")
                if npc and npc.get("longitudinal_gap_m", 0) < -5.0:
                    passed_lead = True
                if passed_lead:
                    lat = abs(r["ego"]["lane_lat"])
                    if lat <= 0.5:
                        returned_lane = True
            overtake_completed = overtake_completed or passed_lead
            returned_to_route_lane = returned_to_route_lane or returned_lane

        if not crashed and not offroad:
            if overtake_completed and returned_to_route_lane:
                success = True
            elif not overtake_completed:
                failure_reason = "failed_to_complete_overtake"
            elif not returned_to_route_lane:
                failure_reason = "failed_to_return_to_route_lane"
        diag = {
            "overtake_completed": overtake_completed,
            "returned_to_route_lane": returned_to_route_lane,
        }

    elif category == "static_full_blockage":
        # Standard: Standstill before obstacle, safe distance in [2m, 15m], no crash, no offroad
        stopped = bool(ep_summary.get("stopped_before_obstacle", False))
        final_gap = ep_summary.get("final_standstill_gap_m")
        if final_gap is None and trajectory_log:
            final_rec = trajectory_log[-1]
            npc = final_rec.get("closest_npc")
            final_gap = npc.get("longitudinal_gap_m") if npc else None
            stopped = final_rec["ego"]["speed_m_s"] < 0.2

        safe_gap = (final_gap is not None and 2.0 <= final_gap <= 15.0)
        if not crashed and not offroad:
            if stopped and safe_gap:
                success = True
            elif not stopped:
                failure_reason = "did_not_come_to_complete_stop"
            else:
                failure_reason = f"standstill_gap_out_of_bounds_{final_gap}m"
        diag = {
            "stopped_before_obstacle": stopped,
            "final_standstill_gap_m": final_gap,
        }

    elif category == "blockage_clearance_resume":
        # Standard: Resumed after blockage cleared, no persistent deadlock
        resumed = bool(ep_summary.get("resumed_after_clearance", False))
        deadlocked = bool(ep_summary.get("deadlocked", False))
        if not crashed and not offroad:
            if resumed and not deadlocked:
                success = True
            elif not resumed:
                failure_reason = "failed_to_resume_driving"
            else:
                failure_reason = "persistent_deadlock_after_resumption"
        diag = {
            "resumed_after_clearance": resumed,
            "deadlocked": deadlocked,
        }

    elif category == "lateral_perturbation_curve_hazard":
        # Standard: Perturbation verified >= 1.0m, model-closed-loop recovered to |lat| < 0.35m
        actual_pert = float(ep_summary.get("actual_perturbation_m", 0.0))
        recovered = bool(ep_summary.get("recovered_to_center", False))
        model_active = bool(ep_summary.get("model_actively_deciding", True))
        if not crashed and not offroad:
            if actual_pert >= 1.0 and model_active and recovered:
                success = True
            elif actual_pert < 1.0:
                failure_reason = f"perturbation_magnitude_insufficient_{actual_pert:.2f}m"
            elif not recovered:
                failure_reason = "failed_to_recover_within_boundary"
        diag = {
            "actual_perturbation_m": actual_pert,
            "recovered_to_center": recovered,
            "model_actively_deciding": model_active,
        }

    return {
        "category": category,
        "success": success,
        "failure_reason": failure_reason,
        "crashed": crashed,
        "offroad": offroad,
        "raw_route_completion": raw_rc,
        "effective_route_completion": round(effective_rc, 4),
        "diagnostics": diag,
    }
