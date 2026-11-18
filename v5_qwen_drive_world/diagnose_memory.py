"""Optional PyTorch dispatcher trace of the exact failing allocation operator."""
import torch
from torch.utils._python_dispatch import TorchDispatchMode


def describe(value):
    if isinstance(value,torch.Tensor):
        return dict(shape=list(value.shape),dtype=str(value.dtype),device=str(value.device))
    if isinstance(value,(list,tuple)):
        return [describe(x) for x in value]
    if isinstance(value,dict):
        return {str(k):describe(v) for k,v in value.items()}
    if isinstance(value,(str,int,float,bool)) or value is None:
        return value
    return type(value).__name__


class TraceAllocationFailure(TorchDispatchMode):
    def __init__(self, report, save):
        super().__init__()
        self.report, self.save = report, save

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        try:
            return func(*args, **kwargs)
        except torch.OutOfMemoryError:
            self.report['allocation_failure_operator'] = dict(operator=str(func),args=describe(args),kwargs=describe(kwargs))
            self.save()
            raise
