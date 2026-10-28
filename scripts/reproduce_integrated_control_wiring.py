"""Native MRE: default IDM ignores requested braking; verified input policy obeys."""
from pathlib import Path
import numpy as np
from env.data_v2 import configure_egl
from env.semantic_bev import make_semantic_env
from env.integrated import make_external_env, checked_step
from mini_wa_jepa.provenance import provenance, write_json
from mini_wa_jepa.visualization import RerunLogger
from dataset.recording import vehicle_state


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    report = dict(**provenance(), seed=4100200, completed=False, cases=[])
    write_json(output/'mre.json', report)
    _, restore = configure_egl(1)
    try:
        for external in (False, True):
            config = dict(map='S', start_seed=4100200, num_scenarios=1, traffic_density=0.,
                          decision_repeat=5, physics_world_step_size=.02, horizon=100, force_destroy=True)
            sim = make_external_env(config) if external else make_semantic_env(overrides=config, semantic_order=True)
            name = 'verified_external' if external else 'default_IDM'
            try:
                sim.reset(seed=4100200); states = []; applied = []
                command = np.array([0., -1.], np.float32)
                with RerunLogger(app_id='control-wiring-mre', spawn=False, save_path=output/(name+'.rrd')) as log:
                    for step in range(10):
                        states.append(vehicle_state(sim.agent))
                        result = checked_step(sim, command) if external else sim.step(command)
                        actual = np.asarray(result[-1]['raw_action'], np.float32); applied.append(actual)
                        log.set_step(step, step*.1); log.log_state(states[-1])
                        log.log_metrics(dict(requested_brake=float(command[1]), applied_throttle=float(actual[1])))
                    final_speed = float(sim.agent.speed)
                states = np.asarray(states); applied = np.asarray(applied)
                np.savez_compressed(output/(name+'.npz'), states=states, applied_action=applied,
                                    requested_action=np.broadcast_to(command, applied.shape))
                case = dict(mode=name, policy=type(sim.engine.get_policy(sim.agent.id)).__name__,
                            mismatched_steps=int(np.any(np.abs(applied-command) > 1e-7, axis=1).sum()),
                            final_speed_m_s=final_speed)
                report['cases'].append(case); write_json(output/'mre.json', report)
            finally:
                sim.close()
        assert report['cases'][0]['mismatched_steps'] > 0
        assert report['cases'][1]['mismatched_steps'] == 0
        assert report['cases'][1]['final_speed_m_s'] < .01
        report['completed'] = True
    finally:
        restore(); write_json(output/'mre.json', report)


if __name__ == '__main__':
    run(Path('artifacts/stage7_integrated/control_wiring_mre_001'))
