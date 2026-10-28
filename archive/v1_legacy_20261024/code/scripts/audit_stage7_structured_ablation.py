"""Audit coefficient-flow delivery using original full expert targets."""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np

from mini_wa_jepa.provenance import provenance,write_json
from mini_wa_jepa.visualization.reports import load_stage_runs
from training.flow_matching import file_sha256


def audit(root,wait=False):
    checks=root/'checks';checks.mkdir(exist_ok=True);outcome=dict(completed=False)
    try:
        started=time.monotonic()
        while True:
            if not (root/'status.json').exists() and wait and time.monotonic()-started<=2400:
                time.sleep(3)
                continue
            state=json.loads((root/'status.json').read_text())
            if state.get('error'):raise RuntimeError(state['error'])
            if state['completed']:break
            if not wait:raise RuntimeError('Experiment incomplete')
            if time.monotonic()-started>2400:raise TimeoutError('Experiment wait exceeded40minutes')
            time.sleep(3)
        source=json.loads((root/'provenance.json').read_text())
        for name in ('models/flow_matcher_structured.py','models/flow_matcher_transformer.py','models/flow_matcher_metric.py',
                     'models/joint.py','scripts/run_stage7_structured_ablation.py','scripts/run_stage7_metric_ablation.py',
                     'dataset/flow_features.py','scripts/report_stage7.py','mini_wa_jepa/normalization.py'):
            if file_sha256(name)!=source['source_files'][name]:raise AssertionError(f'Training source changed: {name}')
        protected=json.loads((root/'protected_checkpoints.json').read_text())
        for file,digest in protected.items():
            if file_sha256(file)!=digest:raise AssertionError(f'Prior checkpoint changed: {file}')
        references=json.loads((root/'protected_reference_files.json').read_text())
        for file,digest in references.items():
            if file_sha256(file)!=digest:raise AssertionError(f'Prior experiment artifact changed: {file}')
        summary=json.loads((root/'summary.json').read_text());cfg=summary['config']
        if cfg['dimensions']==20:
            name='scripts/run_stage7_structured20_ablation.py'
            if file_sha256(name)!=source['source_files'][name]:raise AssertionError('20D launcher changed')
        original_cohort=json.loads((Path(cfg['baseline_root'])/'cohort.json').read_text())
        if original_cohort!=json.loads((root/'cohort.json').read_text()):raise AssertionError('Cohort changed')
        with np.load(root/'trajectory_basis.npz') as b:
            if b['basis'].shape!=(40,cfg['dimensions']):raise AssertionError('Basis dimensions')
            np.testing.assert_allclose(b['basis'].T@b['basis'],np.eye(cfg['dimensions']),atol=1e-6)
        for run_id,run in summary['runs'].items():
            history=[json.loads(s) for s in (root/run_id/'history.jsonl').read_text().splitlines()]
            if len(history)!=25 or history[-1]['global_step']!=7825:raise AssertionError('Training budget mismatch')
            if not 15e6<=run['full_policy_parameters']<=30e6:raise AssertionError('Parameter budget')
            for tau,m in run['validation']['temperatures'].items():
                with (np.load(root/run_id/'validation'/f'tau_{float(tau):g}.npz') as raw,
                      np.load(Path(cfg['baseline_root'])/f'metric_{run["seed"]}/validation/tau_{float(tau):g}.npz') as old):
                    if raw['candidates'].shape!=(2048,8,10,4):raise AssertionError('Wrong prediction shape')
                    if not np.isfinite(raw['candidates']).all():raise AssertionError('Nonfinite prediction')
                    for name in ('target','mse','anchors','episodes'):
                        if not np.array_equal(raw[name],old[name]):raise AssertionError(f'Original labels changed: {name}')
                    error=np.linalg.norm(raw['candidates'][:,0,:,:2]-raw['target'][...,:2],axis=-1)
                    if not np.isclose(error.mean(),m['ade_m'],atol=1e-7):raise AssertionError('ADE mismatch')
                    if not np.isclose(error[:,-1].mean(),m['fde_m'],atol=1e-7):raise AssertionError('FDE mismatch')
                    i,j=np.triu_indices(8,1)
                    diversity=np.linalg.norm(raw['candidates'][:,i,:,:2]-raw['candidates'][:,j,:,:2],axis=-1).mean()
                    if not np.isclose(diversity,m['diversity_m'],atol=1e-7):raise AssertionError('Diversity mismatch')
        def command(name,args):
            with (checks/f'{name}.log').open('w') as stream:
                p=subprocess.run(args,stdout=stream,stderr=subprocess.STDOUT)
            if p.returncode:raise RuntimeError(f'{name} failed; see checks/{name}.log')
        command('report',[sys.executable,'-m','scripts.report_stage7_structured_ablation','--root',str(root)])
        native=[]
        for i,path in enumerate(sorted(root.glob('*/samples/**/*.rrd'))):
            command(f'rrd_{i:03}',[sys.executable,'-m','rerun','rrd','print',str(path)])
            native.append(dict(path=str(path),sha256=file_sha256(path)))
        if len(native)!=24:raise AssertionError('Expected24periodicRRD')
        write_json(checks/'rrd_checks.json',dict(passed=True,files=native))
        rows,_,errors=load_stage_runs(Path.cwd(),7)
        added=[r for r in rows if r.get('evaluation_kind')=='validation_structured_ablation'
               and r['run_id'].startswith(f'structured{cfg["dimensions"]}_v2_')]
        if errors or len(added)!=8:raise AssertionError(f'Dashboard: {len(added)} rows; {errors}')
        command('marimo_check',['.venv/bin/marimo','check','scripts/dashboard_marimo.py'])
        command('dashboard_export',['.venv/bin/marimo','export','html','scripts/dashboard_marimo.py','-o',str(root/'dashboard.html')])
        text=(checks/'tests.log').read_text();match=re.search(r'Ran (\d+) tests',text)
        if not match or int(match[1])<124 or not text.rstrip().endswith('OK'):raise AssertionError('Full suite has not passed')
        stdout=Path('/tmp/mini-wa-jepa-structured20-v2-001.log' if cfg['dimensions']==20 else '/tmp/mini-wa-jepa-structured-v2-001.log')
        if stdout.exists():shutil.copyfile(stdout,root/'training_stdout.log')
        resource=[]
        for path in sorted(root.glob('*/resources.jsonl')):
            records=[json.loads(s) for s in path.read_text().splitlines()]
            resource.append(dict(run=path.parent.name,epoch_samples=len(records),
                max_allocator_bytes=max(r['peak_allocator_bytes'] for r in records),
                max_rss_bytes=max(r['rss_bytes'] for r in records),max_pss_bytes=max(r['pss_bytes'] for r in records),
                min_available_ram_bytes=min(r['available_ram_bytes'] for r in records)))
        write_json(root/'resources_summary.json',dict(runs=resource))
        outcome.update(completed=True,tests=int(match[1]),native_rrd_files=len(native),dashboard_rows=len(added),
            protected_checkpoints=len(protected),protected_checkpoints_unchanged=True,
            protected_reference_files=len(references),protected_reference_files_unchanged=True,
            critical_source_unchanged=True,raw_metrics_and_original_labels_verified=True,
            validation_only=True,execution_backend='Native ROCm / SDPA MATH')
        write_json(checks/'delivery.json',outcome)
        command('report_final',[sys.executable,'-m','scripts.report_stage7_structured_ablation','--root',str(root)])
        write_json(root/'reporting_provenance.json',provenance())
        print(json.dumps(outcome),flush=True)
    except BaseException:
        outcome['error']=traceback.format_exc();write_json(checks/'delivery.json',outcome);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--wait',action='store_true')
    a=p.parse_args();audit(a.root,a.wait)
