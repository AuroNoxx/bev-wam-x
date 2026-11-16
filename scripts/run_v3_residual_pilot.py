"""Bounded two-arm pilot. Trainer retains checkpoint ownership; no test reads."""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/diagnostics/v3-residual-pilot'


def run_logged(command, logfile, phase, arm):
    state = {'phase': phase, 'arm': arm, 'command': command, 'log': str(logfile), 'started_unix': time.time()}
    with logfile.open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        state['pid'] = process.pid
        (OUT / 'pilot-status.json').write_text(json.dumps(state, indent=2))
        seen = set()
        while True:
            lines = logfile.read_text(encoding='utf-8', errors='replace').splitlines()
            for line in lines:
                if line.startswith('EPOCH '):
                    row = json.loads(line[6:]); epoch = row['epoch']
                    if epoch not in seen:
                        seen.add(epoch)
                        print(json.dumps({'arm': arm, 'epoch': epoch, 'budget_epochs': 10, 'step': row['step'],
                                          'meanADE8_m': row.get('trajectory/meanADE8_m'),
                                          'validation_loss': row.get('validation_loss'), 'epoch_seconds': row['epoch_seconds']}), flush=True)
            if process.poll() is not None: break
            time.sleep(1)
    state.update(exit_code=process.returncode, finished_unix=time.time())
    (OUT / 'pilot-status.json').write_text(json.dumps(state, indent=2))
    if process.returncode:
        raise RuntimeError(f'{phase} {arm} failed; see {logfile}')


def main():
    assert json.loads((OUT / 'parameterization-tests.json').read_text())['pass']
    assert json.loads((OUT / 'resume-test/result.json').read_text())['pass']
    for arm in ['absolute', 'cv_residual']:
        run = ROOT / ('artifacts/lightning/flow-v3-' + arm + '-pilot10')
        if (run / 'latest.ckpt').exists(): raise RuntimeError('Existing pilot run; do not silently overwrite or restart')
    for arm in ['absolute', 'cv_residual']:
        print('START_ARM', arm, flush=True)
        command = [sys.executable, '-u', 'scripts/train_lightning.py', '--settings',
                   'configs/lightning-stage1-v3-' + arm + '-pilot.json', '--run-dir',
                   'artifacts/lightning/flow-v3-' + arm + '-pilot10', '--max-steps', '6620']
        run_logged(command, ROOT / ('logs/v3-pilot-' + arm + '.log'), 'training', arm)
        print('EVALUATING', arm, flush=True)
        run_logged([sys.executable, '-u', 'scripts/evaluate_v3_pilot.py', '--arm', arm],
                   ROOT / ('logs/v3-pilot-' + arm + '-evaluation.log'), 'evaluation', arm)
        print('ARM_COMPLETE', arm, flush=True)
    (OUT / 'pilot-status.json').write_text(json.dumps({'phase': 'completed', 'arms': ['absolute', 'cv_residual'],
                                                     'epochs_each': 10, 'steps_each': 6620}, indent=2))
    print('PILOT_COMPLETE', flush=True)


if __name__ == '__main__': main()
