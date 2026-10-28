"""External-control environment and per-step actuation attestation."""
import numpy as np
from env.semantic_bev import make_semantic_env


def make_external_env(overrides):
    from metadrive.policy.env_input_policy import EnvInputPolicy
    config = dict(overrides, agent_policy=EnvInputPolicy, action_check=True)
    return make_semantic_env(overrides=config, semantic_order=True)


def checked_step(sim, action):
    from metadrive.policy.env_input_policy import EnvInputPolicy
    policy = sim.engine.get_policy(sim.agent.id)
    if type(policy) is not EnvInputPolicy:
        raise RuntimeError('External control is not connected: ego policy is not EnvInputPolicy')
    action = np.asarray(action, np.float32)
    if action.shape != (2,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1):
        raise ValueError('Expected finite bounded steering/throttle command')
    result = sim.step(action)
    raw = result[-1].get('raw_action')
    if raw is None:
        raise RuntimeError('Simulator did not report the actually applied raw_action')
    if (not np.allclose(raw, action, rtol=0, atol=1e-7) or
            not np.allclose(sim.agent.current_action, action, rtol=0, atol=1e-7)):
        raise RuntimeError('Requested command differs from actual policy/vehicle actuation')
    return result
