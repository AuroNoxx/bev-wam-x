"""Demonstration eligibility separates causal observations from noisy targets."""
import numpy as np


def eligible_navigation_anchors(valid, route_valid, events, noise_active=None, horizon_steps=20):
    valid, route_valid, events = map(np.asarray, (valid, route_valid, events))
    n = len(valid)
    if route_valid.shape != (n, 8) or events.shape[0] != n:
        raise ValueError('Episode field length mismatch')
    # Collector EVENTS contract: aggregate crash at0, native off-road at4.
    bad = events[:, 0].astype(bool) | events[:, 4].astype(bool)
    if noise_active is not None:
        if np.shape(noise_active) != (n,):
            raise ValueError('Noise mask shape mismatch')
        bad |= np.asarray(noise_active, bool)
    prefix = np.r_[0, np.cumsum(bad)]
    steps = np.arange(n)
    invalid_future = prefix[np.minimum(steps+horizon_steps, n)]-prefix[steps]
    return valid.all(1) & (route_valid.sum(1) >= 2) & (invalid_future == 0)
