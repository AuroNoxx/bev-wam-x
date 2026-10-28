"""Finish delivery after the isolated loss experiment; preserve failures verbatim."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np

from mini_wa_jepa.provenance import provenance, write_json
from mini_wa_jepa.visualization.reports import load_stage_runs
from scripts.report_stage7_metric_ablation import make_report
from training.flow_matching import file_sha256


def audit(root, wait=False):
    checks = root / 'checks'; checks.mkdir(exist_ok=True)
    outcome = dict(completed=False)
    try:
        started = time.monotonic()
        while True:
            state = json.loads((root / 'status.json').read_text())
            if state.get('error'): raise RuntimeError(state['error'])
            if state['completed']: break
            if not wait: raise RuntimeError('Experiment incomplete')
            if time.monotonic() - started > 2400: raise TimeoutError('Experiment wait exceeded 40 minutes')
            time.sleep(3)
        frozen = json.loads((root / 'provenance.json').read_text())
        critical = ['models/flow_matcher_metric.py','models/flow_matcher_transformer.py',
                    'models/joint.py','scripts/run_stage7_metric_ablation.py','dataset/flow_features.py',
                    'mini_wa_jepa/normalization.py','scripts/report_stage7.py']
        for name in critical:
            if file_sha256(name) != frozen['source_files'][name]:
                raise AssertionError(f'Training/evaluation source changed during run: {name}')
        protected = json.loads((root / 'protected_checkpoints.json').read_text())
        for path, digest in protected.items():
            if file_sha256(path) != digest: raise AssertionError(f'Prior checkpoint changed: {path}')
        result = make_report(root)
        full = json.loads((root / 'summary.json').read_text())
        for run_id, run in full['runs'].items():
            history = [json.loads(line) for line in (root / run_id / 'history.jsonl').read_text().splitlines()]
            if len(history) != 25 or history[-1]['global_step'] != 25 * 313:
                raise AssertionError('Wrong matched training budget')
            for tau, metrics in run['validation']['temperatures'].items():
                with np.load(root / run_id / 'validation' / f'tau_{float(tau):g}.npz') as raw:
                    if raw['candidates'].shape != (2048,8,10,4): raise AssertionError('Wrong evaluation cohort')
                    if not np.isfinite(raw['candidates']).all(): raise AssertionError('Nonfinite candidate')
                    error = np.linalg.norm(raw['candidates'][:,0,:,:2] - raw['target'][...,:2], axis=-1)
                    if not np.isclose(error.mean(), metrics['ade_m'], atol=1e-7): raise AssertionError('ADE mismatch')
                    if not np.isclose(error[:,-1].mean(), metrics['fde_m'], atol=1e-7): raise AssertionError('FDE mismatch')
        native = []
        for index, path in enumerate(sorted(root.glob('*/samples/**/*.rrd'))):
            logfile = checks / f'rrd_{index:03}.log'
            with logfile.open('w') as stream:
                process = subprocess.run([sys.executable,'-m','rerun','rrd','print',str(path)],
                                         stdout=stream,stderr=subprocess.STDOUT)
            if process.returncode: raise RuntimeError(f'Native RRD readback failed: {path}; see {logfile}')
            native.append(dict(path=str(path),sha256=file_sha256(path),log=str(logfile)))
        if len(native) != 48: raise AssertionError(f'Expected 48 periodic RRD, found {len(native)}')
        write_json(checks / 'rrd_checks.json', dict(passed=True, files=native))
        rows, _, errors = load_stage_runs(Path.cwd(),7)
        added = [r for r in rows if r.get('evaluation_kind') == 'validation_loss_ablation']
        if errors or len(added) != 16: raise AssertionError(f'Dashboard parse: {len(added)} rows, {errors}')
        with (checks / 'dashboard_export.log').open('w') as stream:
            export = subprocess.run(['.venv/bin/marimo','export','html','scripts/dashboard_marimo.py',
                                     '-o',str(root/'dashboard.html')],stdout=stream,stderr=subprocess.STDOUT)
        if export.returncode: raise RuntimeError('Dashboard runtime export failed')
        tests = (checks / 'tests.log').read_text()
        if 'Ran 116 tests' not in tests or not tests.rstrip().endswith('OK'):
            raise AssertionError('Missing successful 116-test full suite')
        log = Path('/tmp/mini-wa-jepa-metric-v2-001.log')
        if log.exists(): shutil.copyfile(log, root / 'training_stdout.log')
        write_json(root / 'reporting_provenance.json', provenance())
        outcome.update(completed=True, tests=116, protected_checkpoints=len(protected),
                       protected_checkpoints_unchanged=True, critical_source_unchanged=True,
                       native_rrd_files=len(native), dashboard_rows=len(added),
                       raw_metric_recalculation_passed=True, validation_only=True,
                       full_policy_parameters=24868978,
                       actual_execution_backend='Native ROCm; SDPA MATH explicitly scoped in runner',
                       hardware_profile_note='Advertised backends/preferences are capabilities, not the executed backend')
        write_json(checks / 'delivery.json', outcome)
        report = root / 'REPORT.md'
        with report.open('a') as stream:
            stream.write(f'\n交付复核：116项测试通过；48个周期RRD原生读回通过；16行新验证指标看板解析及运行时HTML导出通过；'
                         f'{len(protected)}个既有best.pt哈希未变；训练核心源码哈希与冻结快照一致。'
                         '硬件profile中的FLASH/EFFICIENT为探测能力，本轮执行显式限定SDPA MATH。\n')
        print(json.dumps(outcome,ensure_ascii=False),flush=True)
    except BaseException:
        outcome['error']=traceback.format_exc(); write_json(checks / 'delivery.json',outcome)
        raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--wait',action='store_true')
    args=parser.parse_args();audit(args.root,args.wait)
