"""Preserve the sampler stream when Lightning resumes at an epoch boundary."""
import copy


def boundary_sampler_state(checkpoint):
    states=checkpoint['loops']['fit_loop']['state_dict'].get('combined_loader',[])
    if len(states)!=1:return None
    state=states[0].get('_index_sampler_state')
    if state and state.get('position')==state.get('num_batches'):
        return copy.deepcopy(state)
    return None


def restore_boundary_sampler(loader,state):
    if state is not None:
        loader.batch_sampler.load_state_dict(state)
    return loader
