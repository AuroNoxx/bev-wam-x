"""Reset the declared development navigation or controlled traffic scenario."""
import numpy as np
from configs.stage2 import BEVConfig
from dataset.recording import vehicle_state
from env.data_v2 import NativeBEVProvider
from env.integrated import make_external_env
from env.traffic_teacher import spawn_slow_actor


def setup_driving_scene(scene):
    controlled = 'case' in scene
    home = min(1, scene['lanes']-1) if controlled else 1
    overrides = dict(map=scene.get('map', 'S'), start_seed=scene['seed'], num_scenarios=1,
        traffic_density=scene.get('density', 0.), horizon=600 if controlled else 800,
        decision_repeat=5, physics_world_step_size=.02, force_destroy=True,
        random_lane_width=False, random_lane_num=False, random_agent_model=False,
        random_spawn_lane_index=False,
        agent_configs={'default_agent':{'spawn_lane_index':('>', '>>', home)}})
    if controlled:
        overrides.update(map=3, map_config=dict(type='block_sequence', config='S', lane_num=scene['lanes'], exit_length=250),
            agent_configs={'default_agent':{'spawn_lane_index':('>>', '>>>', home), 'spawn_longitude':5.}})
    sim = make_external_env(overrides)
    try:
        obs, _ = sim.reset(seed=scene['seed']); npcs = []
        if controlled:
            lanes = sim.current_map.road_network.graph['>>']['>>>']
            is_stationary = bool(scene.get('stationary', scene.get('case') in ('blocked_static', 'static_blockage')))
            npcs = [spawn_slow_actor(sim, lanes[i], 5.+scene['lead_distance_m'], scene['seed']+j, stationary=is_stationary)
                    for j, i in enumerate(scene['actor_lanes'])]
            before = np.array([vehicle_state(v) for v in [sim.agent, *npcs]])
            sim.engine.taskMgr.step(); sim.engine.taskMgr.step()
            np.testing.assert_array_equal(before, [vehicle_state(v) for v in [sim.agent, *npcs]])
        provider = NativeBEVProvider(sim, BEVConfig()); provider.reset(None if controlled else obs)
        return sim, provider, [v.id for v in npcs], overrides
    except BaseException:
        sim.close(); raise
