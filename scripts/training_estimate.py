"""Compute measured-throughput estimates, never a claim of convergence time."""
from pathlib import Path
import json,math
ROOT=Path(__file__).resolve().parents[1]
def estimate():
    path=ROOT/'data/occ_v1/manifest.json'
    dataset=json.loads(path.read_text()) if path.exists() else None
    cfg=json.loads((ROOT/'configs/train_v1.json').read_text());results={}
    for mode in ['flow','world']:
        speed=json.loads((ROOT/f'artifacts/v1/v1-{mode}-speed/speed.json').read_text())
        n=dataset['counts']['train'] if dataset else 1000
        v=dataset['counts']['validation'] if dataset else 200
        train_seconds=math.ceil(n/speed['batch'])*speed['mean_step_seconds']
        val_seconds=speed['validation_seconds']*v/speed['validation_windows']
        total=(train_seconds+val_seconds)*1.25  # Explicit assumption for save/I/O/monitoring overhead, not a measurement.
        results[mode]={'train_windows':n,'validation_windows':v,'actual_manifest':bool(dataset),'samples_s':speed['samples_per_second'],
                       'parameters':speed['parameters'],'allocated_gib':speed['peak_allocated_gib'],'reserved_gib':speed['peak_reserved_gib'],
                       'epoch_seconds_with_assumed_25pct_overhead':total,'15_epoch_minutes':total*15/60,'60_epoch_minutes':total,
                       'note':'15 minimum / 60 maximum epochs with patience=10; actual stop depends on held-out validation. I/O on larger corpus may be slower.'}
    (ROOT/'artifacts/v1/time-estimate.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    return results
if __name__=='__main__':print(json.dumps(estimate(),indent=2))
