"""Finish prespecified experiments after collectors finish; no automatic retuning."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
from mini_wa_jepa.provenance import write_json


def run(root):
    state=dict(completed=False,jobs=[],started=time.time());status=root/'remaining_status.json'
    def job(name,module,args,marker):
        state['current']=name;write_json(status,state)
        if marker.exists():
            m=json.loads(marker.read_text())
            if not m.get('completed'):raise RuntimeError(f'Preserve incomplete output: {marker}')
            state['jobs'].append(dict(name=name,reused_completed=True));write_json(status,state);return
        command=[sys.executable,'-m',module,*map(str,args)]
        with (root/f'{name}.log').open('w') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        state['jobs'].append(dict(name=name,command=command,returncode=result.returncode));write_json(status,state)
        if result.returncode:raise RuntimeError(f'{name} failed; inspect preserved log')
    try:
        state['current']='waiting_for_collection';write_json(status,state)
        plan_path=root/'collection_plan.json'
        plan=json.loads(plan_path.read_text()) if plan_path.exists() else {}
        parts=[Path(p) for p in plan['parts']] if plan else [root/name for name in ('risk_train_a','risk_train_b','risk_val','risk_test')]
        while True:
            manifests=[json.loads((p/'manifest.json').read_text()) if (p/'manifest.json').exists() else {} for p in parts]
            if any(m.get('error') for m in manifests):raise RuntimeError('Collector failed; no context silently removed')
            if all(m.get('completed') for m in manifests):break
            time.sleep(5)
        for previous in plan.get('motion_reuse',[]):
            state['current']='waiting_for_motion_recovery';write_json(status,state)
            while True:
                path=Path(previous)/'manifest.json'
                recovery=json.loads(path.read_text()) if path.exists() else {}
                if recovery.get('error'):raise RuntimeError(f'Motion recovery failed: {previous}')
                if recovery.get('completed'):break
                time.sleep(5)
        job('recover_motion','scripts.recover_stage7_motion',
            ['--inputs',*parts,'--output',root/'risk_motion_001',
             *(['--reuse',*plan['motion_reuse']] if plan.get('motion_reuse') else [])],root/'risk_motion_001/manifest.json')
        job('risk_features','scripts.cache_stage7_risk_features',
            ['--inputs',*parts,'--motion',root/'risk_motion_001','--output',root/'risk_features_001',
             *(['--collection-audit',plan['source_audit']] if plan.get('source_audit') else [])],root/'risk_features_001/manifest.json')
        runs=root/'risk_runs'
        for seed in (20261021,20261022):
            for kind in ('world','zero'):
                name=f'{kind}_{seed}';args=['--data',root/'risk_features_001','--output',runs/name,'--seed',seed]
                if kind=='zero':args.append('--zero-world')
                job(name,'scripts.train_stage7_risk',args,runs/name/'report.json')
        job('risk_test_evaluation','scripts.evaluate_stage7_risk',
            ['--data',root/'risk_features_001','--runs',runs,'--output',root/'risk_eval'],root/'risk_eval/report.json')
        # Repeat the frozen Flow evaluation with complete provenance and idle collectors.
        # Compare raw predictions to the earlier run rather than selecting the better result.
        for seed in (20261021,20261022):
            job(f'flow_eval_final_{seed}','scripts.evaluate_flow_matching',
                ['--data',root/'flow_features_001','--run',root/f'flow_{seed}','--output',root/f'flow_eval_final_{seed}'],
                root/f'flow_eval_final_{seed}/summary.json')
        job('closed_smoke_full','scripts.evaluate_stage7_closed_loop',
            ['--arm','full','--seed',20261021,'--smoke','--output',root/'closed_smoke_full_001'],
            root/'closed_smoke_full_001/report.json')
        for seed in (20261021,20261022):
            for arm in ('mse','geometry','full'):
                name=f'closed_{arm}_{seed}'
                job(name,'scripts.evaluate_stage7_closed_loop',['--arm',arm,'--seed',seed,'--output',root/name],
                    root/name/'report.json')
        state.update(completed=True,current='experiments_complete',elapsed_s=time.time()-state['started'])
    except BaseException:
        state['error']=traceback.format_exc();raise
    finally:write_json(status,state)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('artifacts/restart_stage7'))
    run(p.parse_args().root)
