"""Deploy the frozen model as a fixed-shape BF16 HIP Graph on gfx1100."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn.utils.fusion import fuse_conv_bn_eval
from torchvision.models.resnet import BasicBlock

from .phase2_data import Corpus
from .phase2_model import GAMMA, Phase2, configure_sdpa
from .render_probe import LOCKED_CONFIG, locked_tier


INPUTS = ('rgb','lidar','nav','cmd')


def load_model(weights):
    checkpoint = torch.load(weights,map_location='cpu',weights_only=True)
    config = checkpoint['config']
    if (config['architecture'] not in ('Phase2', 'Phase26') or config['gamma']!=GAMMA or config['renderer']!=locked_tier()
            or config['renderer_sha256']!=hashlib.sha256(LOCKED_CONFIG.read_bytes()).hexdigest()):
        raise ValueError('Checkpoint and live renderer or photometric contract differ')
    model = Phase2(
        temporal=bool(config.get('temporal', False)),
        nav_rich=bool(config.get('nav_rich', False)),
    )
    model.load_state_dict(checkpoint['model'],strict=True)
    return model.eval()


def fold_for_inference(model):
    if model.training:
        raise ValueError('Use evaluation BatchNorm statistics')
    count = 0
    for module in list(model.modules()):
        if isinstance(module,BasicBlock):
            for first, second in (('conv1','bn1'),('conv2','bn2')):
                setattr(module,first,fuse_conv_bn_eval(getattr(module,first),getattr(module,second)))
                setattr(module,second,nn.Identity())
                count += 1
        if isinstance(module,nn.Sequential):
            for i in range(len(module)-1):
                if isinstance(module[i],nn.Conv2d) and isinstance(module[i+1],nn.BatchNorm2d):
                    module[i] = fuse_conv_bn_eval(module[i], module[i+1])
                    module[i+1] = nn.Identity()
                    count += 1
    return count


def model_input(sample):
    if 'front' in sample:
        return tuple(torch.from_numpy(np.ascontiguousarray(x))[None] for x in
                     (sample['front'],*sample['sides'],sample['lidar'],sample['nav'],sample['cmd']))
    return tuple(torch.from_numpy(np.ascontiguousarray(sample[k]))[None] for k in INPUTS)


class NativeCDriver(nn.Module):
    def __init__(self,model):
        super().__init__()
        self.model=model

    def forward(self,front,left,right,lidar,nav,cmd):
        return self.model((front,left,right),lidar,nav,cmd)


class Stage26Driver(nn.Module):
    """Fixed-shape 11-tensor wrapper for Stage 2.6 BF16 HIP Graph capture and replay."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, front, left, right, lidar, nav, cmd, route_inst, route_dist,
                prev_map_tokens, ego_delta, has_prev):
        return self.model(
            (front, left, right), lidar, nav, cmd,
            route_inst=route_inst, route_dist=route_dist,
            prev_map_tokens=prev_map_tokens, ego_delta=ego_delta, has_prev=has_prev,
        )


class Inference:
    def __init__(self,model,example,graph):
        self.model = model
        self.pinned = tuple(x.pin_memory() for x in example)
        self.static = tuple(x.to('cuda',non_blocking=True) for x in self.pinned)
        self.graph = None
        self.output = None
        if graph:
            stream = torch.cuda.Stream()
            current = torch.cuda.current_stream()
            stream.wait_stream(current)
            with torch.cuda.stream(stream), torch.inference_mode(), torch.autocast('cuda',torch.bfloat16,cache_enabled=False):
                for _ in range(5):
                    model(*self.static)
            current.wait_stream(stream)
            torch.cuda.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.inference_mode(), torch.cuda.graph(self.graph), torch.autocast('cuda',torch.bfloat16,cache_enabled=False):
                self.output = model(*self.static)
        self.host_output = (
            {k: torch.empty(v.shape, dtype=torch.float32, pin_memory=True) for k, v in self.output.items()}
            if self.output is not None else None
        )

    @torch.inference_mode()
    def __call__(self, *observation):
        for pinned, device, value in zip(self.pinned,self.static,observation):
            pinned.copy_(value)
            device.copy_(pinned,non_blocking=True)
        if self.graph is not None:
            self.graph.replay()
            return self.output
        with torch.autocast('cuda',torch.bfloat16):
            return self.model(*self.static)

    @torch.inference_mode()
    def run_cpu(self, *observation):
        out = self(*observation)
        if self.host_output is None:
            self.host_output = {k: torch.empty(v.shape, dtype=torch.float32, pin_memory=True) for k, v in out.items()}
        for k, v in out.items():
            self.host_output[k].copy_(v if v.dtype == torch.float32 else v.float(), non_blocking=True)
        torch.cuda.synchronize()
        return self.host_output


@torch.inference_mode()
def benchmark(weights,dataset,cache,out,iterations,require_hip_graph=False):
    configure_sdpa('aotriton')
    torch.cuda.tunable.tuning_enable(False)
    torch.set_num_threads(3)
    val = Corpus(dataset,cache,'val')
    observations = [model_input(val[i]) for i in (10, 250, 500, 750, 1000, 1500, 1990)]
    native=len(observations[0])==6
    def deployed():
        base=load_model(weights)
        return (NativeCDriver(base) if native else base).eval().cuda().to(memory_format=torch.channels_last)
    model = deployed()
    with torch.autocast('cuda',torch.bfloat16):
        plain = model(*(x.cuda() for x in observations[0]))
    folded_pairs = fold_for_inference(model)
    with torch.autocast('cuda',torch.bfloat16):
        folded = model(*(x.cuda() for x in observations[0]))
    folds = {k: float((v.float()-folded[k].float()).abs().max()) for k,v in plain.items()}
    if not all(torch.isfinite(x).all() for x in plain.values()):
        raise RuntimeError('Nonfinite evaluation result from the frozen checkpoint')
    fold_near = float((plain['path'][:,:16]-folded['path'][:,:16]).abs().max())
    if not all(torch.isfinite(x).all() for x in folded.values()) or fold_near>.12 or folds['path']>.25 or folds['lines']>.35:
        model = deployed()
        folded_pairs = 0
    del plain,folded
    results = {}; expected = []
    for name in ('eager','hip_graph'):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            engine = Inference(model,observations[0],graph=name=='hip_graph')
        except RuntimeError as error:
            if name!='hip_graph':
                raise
            results[name] = dict(error=str(error))
            continue
        for i in range(20):
            engine(*observations[i%len(observations)])
            torch.cuda.synchronize()
        if name=='eager':
            for row in observations:
                expected.append({k:v.cpu().clone() for k,v in engine(*row).items()})
        else:
            for row,golden in zip(observations,expected):
                actual = engine(*row)
                for key in golden:
                    torch.testing.assert_close(actual[key].float().cpu(),golden[key].float(),atol=.002,rtol=.002)
        device_ms,wall_ms = [],[]
        start_event,end_event = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        for i in range(iterations):
            start = time.perf_counter()
            start_event.record()
            output = engine(*observations[i%len(observations)])
            end_event.record()
            output['path'].float().cpu()
            output['occ'].float().cpu()
            wall_ms.append(1000*(time.perf_counter()-start))
            device_ms.append(start_event.elapsed_time(end_event))
        results[name] = dict(iterations=iterations,p50_device_ms=float(np.median(device_ms)),
            p95_device_ms=float(np.percentile(device_ms,95)),p95_wall_ms=float(np.percentile(wall_ms,95)),
            max_allocated_gib=torch.cuda.max_memory_allocated()/2**30,max_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        del output,engine
    if require_hip_graph:
        if 'p95_wall_ms' not in results['hip_graph']:
            raise RuntimeError(f'Locked HIP Graph deployment failed: {results["hip_graph"]}')
        selection = 'hip_graph'
    else:
        selection = min((name for name in results if 'p95_wall_ms' in results[name]),
                        key=lambda k:results[k]['p95_wall_ms'])
    result = dict(selected=selection, results=results, batch=1, folded_batch_norm=folded_pairs,
                  fold_bf16_max_abs=folds,fold_bf16_path_near_max_abs=fold_near,
                  migraphx='not installed on this host')
    out.mkdir(parents=True,exist_ok=True)
    (out/'latency.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    with weights.open('rb') as stream:
        weights_digest = hashlib.file_digest(stream,'sha256').hexdigest()
    export = dict(weights=str(weights.resolve()),weights_sha256=weights_digest,selected=selection,
                  precision='BF16 AMP',attention='AOTriton',gamma=GAMMA,
                  input_shapes=dict(zip(('front','front_left','front_right','lidar','nav','cmd') if native else INPUTS,[list(x.shape) for x in observations[0]])),
                  graph='Captures fixed storage in the process that runs inference')
    (out/'deployment.json').write_text(json.dumps(export,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


def verify_compiler(weights,out):
    configure_sdpa('aotriton')
    torch.cuda.tunable.tuning_enable(False)
    decoder = load_model(weights).decoder.cuda()
    dtype = torch.float32
    torch.manual_seed(81)
    query = torch.randn(2,8,256,device='cuda',dtype=dtype)
    memory = torch.randn(2,256,256,device='cuda',dtype=dtype)

    def forward(q,kv):
        for layer in decoder:
            q = layer(q,kv)
        return q

    def execute(fn):
        q = query.detach().clone().requires_grad_()
        kv = memory.detach().clone().requires_grad_()
        with torch.autocast('cuda',torch.bfloat16):
            value = fn(q,kv)
        grad = torch.autograd.grad(value.float().square().mean(),(q,kv))
        torch.cuda.synchronize()
        return (value.detach(), *grad)

    expected = execute(forward)
    results = {}
    for backend in ('inductor','aot_eager'):
        try:
            actual = execute(torch.compile(forward,backend=backend,dynamic=False,fullgraph=True))
            relative = [float((x.float()-y.float()).norm()/y.float().norm()) for x,y in zip(actual,expected)]
            if max(relative)>.025 or not all(torch.isfinite(t).all() for t in actual):
                raise RuntimeError(f'Compiled decoder numerics exceeded tolerance: {relative}')
            results[backend] = dict(relative_rms_output_and_grads=relative,passed=True)
            break
        except Exception as error:
            results[backend] = dict(passed=False,error=f'{type(error).__name__}: {error}')
            torch._dynamo.reset()
    out.mkdir(parents=True,exist_ok=True)
    (out/'compile.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps(results,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('weights','out'):
        parser.add_argument(f'--{key}',type=Path,required=True)
    parser.add_argument('--dataset',type=Path)
    parser.add_argument('--cache',type=Path)
    parser.add_argument('--iterations',type=int,default=250)
    parser.add_argument('--compile-audit',action='store_true')
    parser.add_argument('--require-hip-graph',action='store_true')
    args=parser.parse_args()
    if args.compile_audit:
        verify_compiler(args.weights,args.out)
    else:
        if args.dataset is None or args.cache is None:
            parser.error('benchmark requires --dataset and --cache')
        benchmark(args.weights,args.dataset,args.cache,args.out,args.iterations,args.require_hip_graph)
