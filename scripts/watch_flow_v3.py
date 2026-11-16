"""Read-only bounded epoch watcher; never changes training or checkpoints."""
import argparse,json,time
from pathlib import Path
import psutil

ROOT=Path(__file__).resolve().parents[1]


def snapshot(after):
    folder=ROOT/'artifacts/lightning/flow-v3'
    log=ROOT/'logs/lightning/flow-v3.log'
    rows=[]
    text=log.read_text(encoding='utf-8',errors='replace')
    for line in text.splitlines():
        if line.startswith('EPOCH '):
            try:rows.append(json.loads(line[6:]))
            except json.JSONDecodeError:pass
    best=float('inf');best_epoch=None;new=[]
    early_best=float('inf');wait_count=0
    for r in rows:
        ade=r['trajectory/meanADE8_m'];improved=ade<best
        if improved:best=ade;best_epoch=r['epoch']
        if ade<early_best-.01:early_best=ade;wait_count=0
        else:wait_count+=1
        if r['epoch']>after:
            new.append({'epoch':r['epoch'],'step':r['step'],'train_loss':r['train_loss_epoch'],
                        'validation_loss':r['validation_loss'],'meanADE_m':ade,
                        'minADE_m':r['trajectory/minADE8_m'],'meanFDE_m':r['trajectory/meanFDE8_m'],
                        'kinematic_residual_mps':r['trajectory/kinematic_residual_mps'],
                        'seconds':r['epoch_seconds'],'new_best_metric':improved,
                        'best_epoch':best_epoch,'best_meanADE_m':best,'early_stop_wait_estimate':wait_count})
    state={};launcher={}
    for path,target in [(folder/'status.json',state),(ROOT/'logs/stage1-v3/stage1-launch.json',launcher)]:
        try:target.update(json.loads(path.read_text()))
        except (FileNotFoundError,json.JSONDecodeError):pass
    runtime=json.loads((folder/'runtime.json').read_text())
    alive=psutil.pid_exists(runtime['pid'])
    stopping_enabled=not runtime.get('early_stopping_disabled_by_explicit_resume_override',False)
    if not stopping_enabled:
        for row in new:row['early_stop_wait_estimate']=None
    return {'new_epochs':new,'last_epoch':rows[-1]['epoch'] if rows else 0,'training_process_alive':alive,
            'early_stopping_enabled':stopping_enabled,
            'status':state.get('status'),'launcher_phase':launcher.get('phase'),
            'best_epoch':best_epoch,'best_meanADE_m':best,
            'log_tail_if_stopped':text.splitlines()[-6:] if not alive else [],
            'checkpoints':[{'name':p.name,'bytes':p.stat().st_size} for p in folder.glob('*.ckpt')]}


def main():
    p=argparse.ArgumentParser();p.add_argument('--after',type=int,required=True);p.add_argument('--wait',type=float,default=40)
    a=p.parse_args();deadline=time.monotonic()+min(a.wait,45)
    while True:
        report=snapshot(a.after)
        if report['new_epochs'] or not report['training_process_alive'] or report['launcher_phase']=='completed' or time.monotonic()>=deadline:
            print(json.dumps(report),flush=True);return
        time.sleep(1)


if __name__=='__main__':main()
