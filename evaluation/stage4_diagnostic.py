"""Pure helpers for a development diagnostic; never a Stage 4 acceptance gate."""
import json
from pathlib import Path
import numpy as np

EVENTS = ('arrive_dest', 'crash', 'crash_vehicle', 'crash_object', 'crash_sidewalk', 'out_of_road', 'max_step')


def scene_suite(suite, manifest, scenario_seed=20261022):
    if suite == 'legacy_four':
        return [dict(id=case, seed=scenario_seed, map='SSS', lanes=lanes, horizon=250,
                     density=0., controlled=True, lead=lead, dynamic=dynamic,
                     lead_distance=distance, category='following' if lead else 'navigation')
                for case, lanes, lead, dynamic, distance in (
                    ('straight_empty', 1, False, False, 0), ('following_slow', 1, True, False, 22),
                    ('lead_stop_resume', 1, True, True, 20), ('multilane_empty', 3, False, False, 0))]
    if suite == 'legacy_six_S':
        return [dict(id=f'legacy_S_{i}', seed=4300000+i, map='S', density=0., horizon=250,
                     controlled=False, category='navigation') for i in range(6)]
    if suite == 'manifest_nav':
        source = json.loads(Path(manifest).read_text())
        if source.get('split') != 'validation' or not source.get('completed'):
            raise ValueError('Completed development validation manifest required')
        scenes = [s for s in source['scenes'] if 'case' not in s][:6]
        if len(scenes) != 6 or any(not 4300000 <= int(s['seed']) < 4400000 for s in scenes):
            raise ValueError('Expected six development navigation seeds; blind seeds forbidden')
        return [dict(s, id=f'nav_{s["map"]}_{s["seed"]}', horizon=800,
                     controlled=False, category='navigation') for s in scenes]
    if suite == 'manifest_nav12':
        source = json.loads(Path(manifest).read_text())
        if source.get('split') != 'validation' or not source.get('completed'):
            raise ValueError('Completed development validation manifest required')
        scenes = [s for s in source['scenes'] if 'case' not in s][:12]
        if len(scenes) != 12 or any(not 4300000 <= int(s['seed']) < 4400000 for s in scenes):
            raise ValueError('Expected twelve development navigation seeds; blind seeds forbidden')
        return [dict(s, id=f'nav_{s["map"]}_{s["seed"]}', horizon=800,
                     controlled=False, category='navigation') for s in scenes]
    raise ValueError(suite)


def scene_overrides(scene):
    home = min(1, scene.get('lanes', 3)-1)
    config = dict(map=scene['map'], start_seed=scene['seed'], num_scenarios=1,
                  traffic_density=scene['density'], horizon=scene['horizon'],
                  decision_repeat=5, physics_world_step_size=.02, force_destroy=True,
                  random_lane_width=False, random_lane_num=False, random_agent_model=False,
                  random_spawn_lane_index=False,
                  agent_configs={'default_agent': {'spawn_lane_index': ('>', '>>', home)}})
    if scene['controlled']:
        config.update(map=3, map_config=dict(type='block_sequence', config='SSS', lane_num=scene['lanes'], exit_length=300),
                      agent_configs={'default_agent': {'spawn_lane_index': ('>>', '>>>', home), 'spawn_longitude': 5.}})
    return config


def recovery_pulse(seed, step, enabled, start_seed=4300000, start=30, period=160, length=6):
    """Recreate the documented collection-time steering pulse; pure helper."""
    active = bool(enabled and step >= start and (step - start) % period < length)
    sign = 1 if (int(seed) + ((int(seed) - int(start_seed)) // 6)) % 2 == 0 else -1
    return active, sign


def causal_ego(speed, heading, previous_heading, dt=.1):
    if dt <= 0:
        raise ValueError('Positive control interval required')
    delta = 0. if previous_heading is None else np.arctan2(np.sin(heading-previous_heading), np.cos(heading-previous_heading))/dt
    return np.array([speed/10., delta], np.float32)


def classify_outcome(category, rows, horizon):
    """Following is diagnostic safety+progress, distinct from destination arrival.

    Missing lead clearance or lead identity is unresolved, never a safety pass.
    This conservative endpoint does not claim full stop/resume or overtaking skill.
    """
    crash = any(any(r['events'].get(k, False) for k in ('crash', 'crash_vehicle', 'crash_object', 'crash_sidewalk')) for r in rows)
    offroad = any(r['events'].get('out_of_road', False) for r in rows)
    arrived = any(r['events'].get('arrive_dest', False) for r in rows)
    gaps = [r.get('tracked_lead_clearance_m') for r in rows]
    complete = len(rows) >= horizon or arrived
    gap_known = bool(gaps) and all(g is not None and np.isfinite(g) for g in gaps)
    progress = float(sum(np.linalg.norm(np.asarray(r['post_state'][:2])-r['state'][:2]) for r in rows))
    safe_follow = (bool(complete and not crash and not offroad and min(gaps) >= 2. and progress > 1.)
                   if category == 'following' and gap_known else None)
    return dict(category=category, arrived=arrived, crash=crash, offroad=offroad,
                completed_declared_observation=complete, route_completion=rows[-1]['route_completion'] if rows else None,
                traveled_m=progress, navigation_success=bool(arrived and not crash and not offroad) if category == 'navigation' else None,
                safe_follow_observation=safe_follow, gap_observation_complete=gap_known if category == 'following' else None,
                minimum_tracked_clearance_m=min(gaps) if gap_known else None,
                stage4_accepted=False)
