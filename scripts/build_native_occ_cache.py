"""Explicit bounded native cache builder; --all requires an additional flag."""
import argparse,json,os
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import torch
torch.set_num_threads(1)
if os.name=='nt':
    import ctypes
    kernel=ctypes.windll.kernel32
    kernel.GetCurrentProcess.restype=ctypes.c_void_p
    kernel.SetPriorityClass.argtypes=[ctypes.c_void_p,ctypes.c_uint32]
    if not kernel.SetPriorityClass(kernel.GetCurrentProcess(),0x4000):raise OSError('cannot set low CPU priority')
from b2d_native_occ import ROOT
from native_occ_cached_data import build_cache

def main():
    p=argparse.ArgumentParser();p.add_argument('--all',action='store_true');p.add_argument('--approve-full-build',action='store_true');p.add_argument('--per-split',type=int,default=3);p.add_argument('--output',default='data/native_occ_aligned_cache');a=p.parse_args()
    if a.all and not a.approve_full_build:raise SystemExit('Full build requires explicit --approve-full-build; do not start without user approval')
    if not a.all and not 1<=a.per_split<=8:raise SystemExit('pilot limited to 1..8 samples per split')
    m=json.loads((ROOT/'data/occ_v3/manifest.json').read_text())
    names=[n for split in ('train','validation') for n in (m['splits'][split] if a.all else m['splits'][split][:a.per_split])]
    print(build_cache(ROOT/a.output,names),flush=True)
if __name__=='__main__':main()


