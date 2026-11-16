from pathlib import Path
import json,math

ROOT=Path(__file__).resolve().parents[1]
reports={name:json.loads((ROOT/f'artifacts/benchmark/{name}-b8/report.json').read_text()) for name in ['flow','world','vision']}
names={'flow':'OCC → Flow Matching trajectory','world':'Action-conditioned OCC forecasting','vision':'Six cameras → compact OCC (frozen backbone)'}
audit=json.loads((ROOT/'data/benchmark_cache/audit.json').read_text())
cleanup=json.loads((ROOT/'logs/self-collected-cleanup-status.json').read_text())
lines=['# Public-data training-cost benchmark','',
       'Measured on this Windows RX 7900 XT 20GB. This is a throughput pilot of explicit small models, not a completed driving policy or a convergence estimate.','',
       '## Actual measurements','',
       '| Module | Total / trainable parameters | Samples/s | Step seconds (batch 8) | PyTorch peak allocated / reserved GiB | Held-out loss |',
       '|---|---:|---:|---:|---:|---:|']
for mode,r in reports.items():
    lines.append(f'| {names[mode]} | {r["parameters"]:,} / {r["trainable_parameters"]:,} | {r["samples_per_second"]:.2f} | {r["mean_step_seconds"]:.4f} | {r["peak_allocated_gib"]:.2f} / {r["peak_reserved_gib"]:.2f} | {r["held_out_loss"]:.4f} |')
lines+=['','Each module: 5 warmup updates, then 200 timed updates. BF16 autocast, FP32 master weights, AdamW, batch 8. Timings include reading/decompressing cached samples, transfer, forward, backward, clipping and optimizer step. Validation/checkpoint costs are reported separately in per-module JSON. Loss values across modules are not comparable.','',
        'All modules showed finite gradients and a nonzero parameter update. The pilot uses only 54 unique training windows and 54 windows from a separate held-out route; repeated passes measure cost, not data scale or generalization. The tiny corpus fits in OS cache, so full-corpus I/O can be slower.','',
        '## Conditional total-time estimates','',
        'All three stages are trained separately. Assume the same number of effective windows per stage and unchanged model, grid, image, and history/horizon settings. 20 epochs is a comparison scenario, NOT an established convergence requirement.','',
        '| Effective windows per stage | All stages, 1 epoch each | All stages, 20 epochs each | With assumed 20–40% scheduling/evaluation/I/O overhead |',
        '|---:|---:|---:|---:|']
estimates=[]
for n in [10000,40000,100000]:
    hours=sum(math.ceil(n/r['batch'])*r['mean_step_seconds'] for r in reports.values())/3600
    estimates.append({'windows_per_stage':n,'hours_one_epoch_each':hours,'hours_20_epochs_each':hours*20,'hours_20_epochs_with_20_to_40_percent_overhead':[hours*24,hours*28]})
    lines.append(f'| {n:,} | {hours:.2f} h | {hours*20:.2f} h | {hours*24:.2f}–{hours*28:.2f} h |')
lines+=['','Formula: sum over stages of ceil(valid_windows / batch_size) × measured_step_seconds × epochs. Real stages may use different dataset sizes and epoch counts. Downloading, conversion, counterfactual data collection, CARLA closed-loop testing and architecture tuning are not included. These are not predictions of time-to-safe-driving.','',
        '## Data actually used','',
        '- Source: `rethinklab/Bench2Drive-V0.0.4`, revision `d84ae18cf20a88b9220df1bc15c073d8824b7ac1`.',
        '- Two HighwayCutIn Town12 clips, different routes/weather. Each has 111 annotation files, but only 110 paired OCC/image frames; unpaired frame 110 is explicitly excluded.',
        '- 4 historical OCC observations spanning 0.6 s. FM: 25 future waypoints at 0.2 s spacing (5 s); world model: future OCC at 1/2/3/4/5 s, conditioned on recorded future ego trajectory and controls.',
        '- Six vehicle-mounted RGB views resized to 448×256 with appropriately scaled projection coordinates. The top-down camera is not a model input.',
        '- Current ego coordinate frame for aligned history; each future ego frame for future OCC. Candidate trajectory specifies the future ego motion. Recorded future controls are training conditions, not available observations for the planner.',
        '- Source OCC grid: 128×128×20, 0.4 m voxels, XY [-25.6,25.6] m and Z [-5,3] m. The benchmark merges each 5 height cells conservatively into 4 height layers and groups semantics into 6 classes.',
        '- Source occupancy zero is named UNFILLED, not certified free space. The downloaded files do not supply a complete visibility/free-space mask. This does not meet a final safety-aware OCC representation by itself.',
        '- The two clips reach only about 47.3 and 39.5 km/h. Their names contain HighwayCutIn, but this is not a 90–120 km/h deployment benchmark.',
        '- UniOcc public probe is downloaded separately (160 continuous OCC frames and paired images); it is not silently mixed into these measured training windows. Its native grid/label conventions differ, so full-size UniOcc timing is NOT established by this report.',
        '- No self-collected frame is used. Downloaded files and local SHA256 values are recorded in `data/public/download-manifest.json`.','',
        '## Model and backend details','',
        '- Vision: ImageNet ResNet18 through layer3, frozen; geometry-based multi-camera feature sampling and a small trainable occupancy decoder. It is not a fine-tuned full ResNet18.',
        '- Planner: compact OCC CNN encoder plus a 4-layer, width-256 trajectory Transformer trained with conditional flow matching.',
        '- World model: compact OCC encoder, conditioning on executed trajectory/controls, and a shared decoder for five future times. No counterfactual prediction quality has been established.',
        '- PyTorch 2.11.0+rocm7.14.0, HIP 7.14.60850. The existing runtime was copied into this project, including a local Python 3.13 runtime. Original environments and drivers were not changed.',
        '- Windows ROCm frozen BF16 BatchNorm inference failed with `miopenStatusUnknownError`. Folding eval BatchNorm into convolutions avoids that operator. CPU equivalence check max absolute error: 8.642673492431641e-7.',
        '- Peak memory above is PyTorch allocator memory, not the entire Windows GPU working set. Do not extrapolate these figures to large dense 3D models or simultaneous simulation.',
        '- Some MIOpen first-use kernels print diagnostics and have substantial startup time. Five warmup updates are excluded from steady-state estimates.','',
        '## Reproduce / inspect','',
        'Run these commands in project PowerShell:','',
        '```powershell',
        r'.\.venv-train\Scripts\python.exe scripts\prepare_public_benchmark.py',
        r'.\.venv-train\Scripts\python.exe scripts\benchmark_training.py --mode flow --batch 8 --warmup 5 --steps 200',
        r'.\.venv-train\Scripts\python.exe scripts\benchmark_training.py --mode world --batch 8 --warmup 5 --steps 200',
        r'.\.venv-train\Scripts\python.exe scripts\benchmark_training.py --mode vision --batch 8 --warmup 5 --steps 200',
        r'.\.venv-train\Scripts\python.exe -m tensorboard.main --logdir logs\tensorboard --host 127.0.0.1 --port 6006',
        '```','',
        'Dashboard: http://127.0.0.1:6006 (local only). Parameters, optimizer states and reports are under `artifacts/benchmark/<mode>-b8/`. Dependencies are recorded in `configs/requirements-training-lock.txt`; ROCm package binaries come from the copied local runtime rather than ordinary PyPI.','',
        '## Cleanup status','',
        f'Self-collection and CARLA were stopped. Automatic approval review rejected deletion of data/raw with "blocked by policy", without a specific reason. {cleanup["remaining_bytes"]/1024**3:.2f} GiB remains and is excluded from training. It has NOT been deleted.','',
        '## Sources','',
        '- https://huggingface.co/datasets/rethinklab/Bench2Drive-V0.0.4',
        '- https://github.com/Thinklab-SJTU/Bench2Drive/blob/0.0.4/tools/occ_online_carla.py',
        '- https://github.com/tasl-lab/UniOcc',
        '- https://download.pytorch.org/models/resnet18-f37072fd.pth','']
(ROOT/'docs/PUBLIC_DATA_BENCHMARK.md').write_text('\n'.join(lines),encoding='utf-8')
(ROOT/'artifacts/benchmark/summary.json').write_text(json.dumps({'measurements':reports,'estimates':estimates,'data_audit':audit},indent=2),encoding='utf-8')
print(json.dumps(estimates,indent=2))
