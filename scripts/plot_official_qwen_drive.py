import sys,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'runtime/qwen-drive-deps'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=R/'artifacts/qwen-drive-official-planner';rows=json.loads((O/'outputs.json').read_text());fig,axes=plt.subplots(2,2,figsize=(12,10));names=['Night intersection / green light','Left turn','Right turn','Slow-down past parked truck']
for ax,r,name in zip(axes.flat,rows,names):
 p=np.array(r['trajectory']);g=np.array(r['ground_truth']);e=np.linalg.norm(p[:,:2]-g[:,:2],axis=1);assert abs(e.mean()-r['ADE_m'])<1e-5
 ax.plot(g[:,1],g[:,0],color='#777777',lw=2,label='Ground truth');ax.plot(p[:,1],p[:,0],color='#176bdb',lw=2,label='Official SFT prediction');ax.scatter([0],[0],color='black',s=35,marker='^',label='Ego start');ax.set_aspect('equal',adjustable='datalim');ax.set_xlabel('Lateral y (m)');ax.set_ylabel('Forward x (m)');ax.grid(alpha=.25);ax.set_title(f"{name}\nADE {e.mean():.3f} m | FDE {e[-1]:.3f} m | {r['seconds']:.2f} s");ax.legend(fontsize=8)
fig.suptitle('Qwen-Drive-1.0-4B + official planner-sft | Direct planning\n4 official demo scenes, 1 trajectory each, 5 seconds at 10 Hz',fontsize=14);fig.tight_layout(rect=(0,0,1,.94));fig.savefig(O/'trajectory_comparison.png',dpi=150)
lines=['# Official Qwen-Drive direct planning test','','Official BF16 VLM + planner-sft. Four official WOD-E2E demos, each 3 views x 4 frames plus ego history/navigation. 1 candidate per scene. No training, world model, Judge, custom head or generated reasoning.','','| Scene | ADE m | FDE m | Seconds | Peak allocated GiB |','|---|---:|---:|---:|---:|']
for r,n in zip(rows,names):lines.append(f"| {n} | {r['ADE_m']:.4f} | {r['FDE_m']:.4f} | {r['seconds']:.3f} | {r['peak_allocated_GiB']:.3f} |")
lines+=['','Mean ADE %.4f m; FDE %.4f m.'%(np.mean([r['ADE_m'] for r in rows]),np.mean([r['FDE_m'] for r in rows])),'','RX7900XT, ROCm7.14, torch2.11, transformers5.14.1 isolated dependencies, SDPA MATH. Official commit 28091c1532e869bc7aee91fc0aef6b3e6fd0b2e0. Both weight SHA256 hashes verified.','','Timing excludes model loading (~29.97s) and CPU image processing, includes GPU transfer/prefill/planning/output transfer. First scene includes first-use overhead. Not a controlled warm benchmark. Peak reserved14.84GiB.','','Selected official demo open-loop only, not safety validation or independent CARLA generalization. Older CARLA recordings only have front view; cannot retroactively supply true side views. Quantized community weights were found but not tested. Full trajectories/ground truth in outputs.json.']
(O/'REPORT.md').write_text('\n'.join(lines),encoding='utf8');print('mean ADE',np.mean([r['ADE_m'] for r in rows]));print('mean FDE',np.mean([r['FDE_m'] for r in rows]))
