"""Continue complete branch partitions through matched training, test and report."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import traceback

from configs.stage6 import Stage6Config
from mini_wa_jepa.provenance import provenance,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('artifacts/restart_stage6'))
    p.add_argument('--partitions',nargs=3,default=['branches_train_a','branches_train_b','branches_val_test'])
    a=p.parse_args();root=a.root;cfg=Stage6Config();repo=Path.cwd()
    status=dict(**provenance(cfg),completed=False,phase='waiting_for_collection',jobs=[])
    env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    def update(phase,**fields):
        status.update(phase=phase,**fields);write_json(root/'suite_status.json',status)
    def run(module,arguments,label):
        command=[sys.executable,'-m',module]+list(map(str,arguments));log_path=root/f'{label}.log'
        with log_path.open('x') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
            status['jobs'].append(dict(command=command,log=str(log_path),pid=child.pid))
            update(label);code=child.wait();status['jobs'][-1]['exit_code']=code;update(label)
        if code:raise RuntimeError(f'{label} exited {code}; see {log_path}')
    try:
        partitions=[root/name for name in a.partitions]
        while True:
            reports=[json.loads((folder/'manifest.json').read_text()) if (folder/'manifest.json').exists() else {} for folder in partitions]
            for folder,report in zip(partitions,reports):
                if report.get('error'):raise RuntimeError(f'Collection failed: {folder}: {report["error"]}')
            if all(r.get('completed') for r in reports):break
            update('waiting_for_collection',collection_contexts=[len(r.get('groups',[])) for r in reports]);time.sleep(10)
        run('scripts.merge_stage6_branches',['--inputs',*partitions,'--output',root/'branches'],'merge')
        run('unittest',['discover','-s','tests','-v'],'tests_final')
        write_json(root/'tests_result.json',dict(passed=True,command='python -m unittest discover -s tests -v',
            log=str(root/'tests_final.log'),test_sources={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in Path('tests').glob('test*.py')}))
        run('scripts.cache_stage6_features',['--data',root/'branches','--output',root/'features'],'feature_cache')
        for seed in cfg.training_seeds:
            for kind in ('conditioned','unconditional'):
                name=f'{kind}_{seed}';folder=root/'runs'/name
                args=['--data',root/'features','--output',folder,'--seed',seed]
                if kind=='unconditional':args+=['--unconditional']
                run('scripts.train_action_conditioned',args,name)
                report=json.loads((folder/'report.json').read_text())
                with tarfile.open(folder/'source_snapshot.tar.gz','w:gz') as tar:
                    for relative,expected_hash in report['source_files'].items():
                        source=repo/relative
                        if hashlib.sha256(source.read_bytes()).hexdigest()!=expected_hash:
                            raise RuntimeError(f'Source changed during training: {relative}')
                        tar.add(source,arcname=relative)
                    for relative in ('GOAL.md','AGENTS.md','docs/STAGE6_PROTOCOL.md'):
                        tar.add(repo/relative,arcname=relative)
        run('scripts.evaluate_action_conditioned',['--data',root/'features','--runs',root/'runs','--output',root/'test_eval'],'test_evaluation')
        run('scripts.report_stage6',['--root',root],'report_generation')
        run('marimo',['export','html','scripts/dashboard_marimo.py','--no-include-code','-o',root/'dashboard.html'],'dashboard_export')
        acceptance=json.loads((root/'acceptance.json').read_text())
        update('report_completed',completed=True,passed=acceptance['passed'],delivery_passed=acceptance['delivery_passed'],scientific_passed=acceptance['scientific_passed'])
    except BaseException:
        update('failed',error=traceback.format_exc());raise


if __name__=='__main__':main()
