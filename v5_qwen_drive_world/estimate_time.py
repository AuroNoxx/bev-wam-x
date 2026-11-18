"""Report measured costs and explicitly conditional epoch estimates."""
import json
from .runtime import ROOT


def summarize(name):
    path=ROOT/'artifacts/v5-qwen-drive-world'/f'accumulation-{name}.json'
    if not path.exists(): return {'status':'not_run'}
    run=json.loads(path.read_text(encoding='utf-8-sig'))
    result={'status':run['phase'],'evidence':str(path.relative_to(ROOT)),
            'scope':run['scope'],'formal_training':False,
            'optimizer_updates':run.get('optimizer_updates',0)}
    if run['phase']!='passed': return result
    rows=run['microbatches'][8:16]
    assert len(rows)==8 and run['optimizer_updates']==2
    update=sum(row['seconds'] for row in rows)
    sample=update/8
    result.update(seconds_per_microbatch=sample,seconds_per_update=update,
                  fit_seconds=run['fit_seconds'],total_seconds=run['seconds'],
                  peak_allocated_GiB=run['peak_allocated_GiB'],peak_reserved_GiB=run['peak_reserved_GiB'],
                  hypothetical_20_epoch_days={str(n):20*n*sample/86400 for n in (1000,7690,10986)})
    monitor=ROOT/'logs/v5-qwen-drive-world'/f'accumulation-{name}-memory.jsonl'
    samples=[json.loads(row) for row in monitor.read_text().splitlines()]
    result['OS_min_available_RAM_GiB']=min(row['available_MB'] for row in samples)/1024
    result['OS_peak_single_adapter_shared_GPU_GiB']=max(g['SharedUsage'] for row in samples for g in row['gpu'])/2**30
    return result


def main():
    flow=summarize('native-cap85-updates2-fla-foreach-tiled')
    joint=summarize('native-joint-cap85-updates2-trace-offload-fla-foreach-tiled-blocks-convtiles-encoder')
    result={'microbatch':1,'accumulation':8,'effective_batch':8,'native_processor_resolution_preserved':True,
            'RGB_Flow':flow,'RGB_Flow_World':joint,
            'old_fallback_seconds_per_sample_approx':55.32,
            'RGB_Flow_speedup_vs_previous_estimate':55.32/flow['seconds_per_microbatch'],
            'formula':'20 * eligible_temporal_samples * measured_seconds_per_microbatch / 86400',
            'limitations':[
                'Each probe repeats one official scene; not actual SFT, not a dataset throughput average.',
                'World probe has one calibrated RGB frame and synthetic zero state/history; no multi-frame joint throughput claim.',
                '7690 counts old raw training frames, not finalized eligible v5 temporal samples.',
                '10986 assumes7690 eligible B2D samples form70percent; required highway data not collected.',
                'EMA future targets, data IO, validation, checkpointing, collection and Stage2 excluded.',
                'Joint diagnostic includes dispatcher tracing; OS adapter shared memory may include mapped pinned memory and other apps.',
                'Original processor sizes are preserved; future1920x1080 highway capture target is not yet benchmarked.'
            ],'formal_training_started':False,'full_training_ETA':None}
    path=ROOT/'artifacts/v5-qwen-drive-world/training-time-estimate-fla.json'
    path.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
