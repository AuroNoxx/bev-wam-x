"""Observed joint diagnostic compute cost, not a full-SFT duration promise."""
import json
from pathlib import Path


def main():
    root=Path(__file__).resolve().parents[1]
    artifacts=root/'artifacts/v5-qwen-drive-world'
    rows=[]
    for batch in [1,2,4]:
        result=json.loads((artifacts/f'lightweight-mb{batch}.json').read_text())
        assert result['phase']=='passed' and result['optimizer_updates']==2
        seconds=result['warm_seconds_per_sample']
        monitor=[json.loads(line) for line in (root/f'logs/v5-qwen-drive-world/lightweight-mb{batch}-memory.jsonl').read_text(encoding='utf-8-sig').splitlines() if line.strip()]
        # Whole-card counter includes runtime allocations, desktop and other apps.
        card_peak=max(g['DedicatedUsage'] for r in monitor for g in r['gpu'])/2**30
        rows.append({'microbatch':batch,'accumulation':8//batch,'effective_batch':8,
                     'seconds_per_sample':seconds,'seconds_per_update':8*seconds,
                     'peak_allocated_GiB':result['peak_allocated_GiB'],
                     'peak_reserved_GiB':result['peak_reserved_GiB'],
                     'whole_card_dedicated_peak_GiB':card_peak,
                     'minimum_available_RAM_MB':min(r['available_MB'] for r in monitor),
                     'assumed_effective_samples':7690,
                     'epoch_hours':{str(e):7690*e*seconds/3600 for e in [1,3,5]}})
    profile=json.loads((artifacts/'profile-before-lightweight.json').read_text())['warm_mean_s']
    report={'scope':'joint diagnostic compute on repeated official three-camera planning scene; original processor pixels, six retained images, 836 tokens',
            'formal_training':False,'selected_microbatch':min(rows,key=lambda r:r['seconds_per_sample'])['microbatch'],
            'profile_llm_backward_fraction':profile['llm_backward_recompute_s']/profile['total_micro_s'],
            'measurements':rows,'full_training_eta_hours':None,
            'excludes':['first compilation and loading','real dataset IO','supervised reason/plan losses','EMA future targets','validation','checkpoint saving','collection','Stage2'],
            'limitations':['7690 is an old raw-frame count, not a validated v5 manifest',
                           'batch members repeat one scene; variable lengths and eight calibrated cameras not timed',
                           'PyTorch peak below16GiB does not mean whole-card dedicated occupancy below16GiB',
                           'two updates prove this execution path, not long-run OOM immunity or driving accuracy']}
    (artifacts/'training-time-estimate-lightweight.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
