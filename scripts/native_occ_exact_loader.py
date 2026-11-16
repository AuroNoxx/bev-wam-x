"""workers=0 StatefulDataLoader resume fix for shared sampler/base-seed RNG.

TorchData's legacy snapshot saves pre-permutation sampler RNG, but restoring it
rewinds the shared generator past DataLoader's base_seed draw. Current epoch is
correct; next epoch otherwise diverges. This wrapper captures actual post-draw
RNG and can reconstruct ONLY the validated no-replacement legacy layout.
Caller MUST independently check checkpoint settings/manifest/full dataset order.
"""
import copy
import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import RandomSampler

class ExactResumeDataLoader(StatefulDataLoader):
    def __init__(self,*args,allow_legacy_resume=False,**kwargs):
        super().__init__(*args,**kwargs)
        if self.num_workers!=0 or not isinstance(self.sampler,RandomSampler) or self.sampler.replacement or self.sampler.num_samples!=len(self.dataset) or self.generator is None or self.sampler.generator is not self.generator:
            raise ValueError('validated mode is workers0, shared explicit generator, no-replacement full RandomSampler')
        self.allow_legacy_resume=allow_legacy_resume;self._restored_generator=None
    def state_dict(self):
        state=super().state_dict()
        state['_exact_generator_v1']=self.generator.get_state().clone()
        return state
    def load_state_dict(self,state):
        state=copy.deepcopy(state)
        if '_exact_generator_v1' in state:
            target=state.pop('_exact_generator_v1')
        else:
            if not self.allow_legacy_resume:raise ValueError('legacy restore requires explicit validated contract')
            expected={'_index_sampler_state','_sampler_iter_state','_sampler_iter_yielded','_num_yielded','_IterableDataset_len_called','_shared_seed','fetcher_state','dataset_state','_iterator_finished'}
            if set(state)!=expected or state['_index_sampler_state'] is not None or state['_shared_seed'] is not None or state['fetcher_state'] is not None:raise ValueError('unsupported legacy loader layout')
            s=state['_sampler_iter_state']
            if set(s)!={'samples_yielded','sampler_iter_state'} or set(s['sampler_iter_state'])!={'yielded','generator'}:raise ValueError('unsupported legacy sampler layout')
            if not 0<=s['samples_yielded']==s['sampler_iter_state']['yielded']<=len(self.dataset):raise ValueError('invalid legacy progress')
            g=torch.Generator();g.set_state(s['sampler_iter_state']['generator'])
            torch.randperm(len(self.dataset),generator=g)
            torch.empty((),dtype=torch.int64).random_(generator=g) # original base_seed draw
            target=g.get_state()
        self._restored_generator=target.clone()
        super().load_state_dict(state)
    def _get_iterator(self):
        restoring=self.next_iter_state is not None
        it=super()._get_iterator()
        if restoring and self._restored_generator is not None:
            self.generator.set_state(self._restored_generator)
            self._restored_generator=None
        return it
