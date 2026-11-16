"""Guarded v3 launcher. No automatic download/rebuild or reuse of v2 weights."""
from pathlib import Path
import argparse,json,subprocess
ROOT=Path(__file__).resolve().parents[1]
PY=ROOT/'.venv-train/Scripts/python.exe'
LOG=ROOT/'logs/stage1-v3'
def main():
    p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');p.add_argument('--finish-max-epochs',action='store_true');args=p.parse_args()
    if args.finish_max_epochs and not args.resume:raise ValueError('--finish-max-epochs requires --resume')
    pause=ROOT/'configs/TRAINING_PAUSED.json'
    if pause.exists() and json.loads(pause.read_text()).get('paused'):
        raise RuntimeError('Training paused for data-contract audit; see configs/TRAINING_PAUSED.json. Existing results are preserved.')
    LOG.mkdir(exist_ok=True,parents=True)
    status=LOG/'stage1-launch.json'
    manifest=ROOT/'data/occ_v3/manifest.json'
    prepared=json.loads(manifest.read_text())
    if prepared.get('preprocess_version')!=3 or prepared.get('occ_file_offset')!=-1 or prepared.get('pilot'):
        raise RuntimeError('Build and verify the independent v3 cache before training')
    if not all(prepared['counts'][s]>0 for s in ['train','validation']):raise RuntimeError('Empty split')
    report={'note':'v3 fresh-run timing not yet measured; no old model initialization'}
    status.write_text(json.dumps({'phase':'launching_training','time_estimate':report}),encoding='utf-8')
    command=[str(PY),'-u',str(ROOT/'scripts/train_lightning.py'),'--mode','flow','--settings','configs/lightning-stage1-v3.json','--run-dir','artifacts/lightning/flow-v3']
    if args.resume:command.append('--resume')
    if args.finish_max_epochs:command.append('--finish-max-epochs')
    training_log=ROOT/'logs/lightning/flow-v3.log';training_log.parent.mkdir(parents=True,exist_ok=True)
    with training_log.open('a',encoding='utf-8') as log:
        process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        status.write_text(json.dumps({'phase':'training','framework':'pytorch-lightning','pid':process.pid,'log':str(training_log),'time_estimate':report}),encoding='utf-8')
        result=process.wait()
    status.write_text(json.dumps({'phase':'completed' if result==0 else 'failed','exit_code':result}),encoding='utf-8')
    if result:raise RuntimeError(f'Training exited {result}; see {training_log}')
if __name__=='__main__':main()
