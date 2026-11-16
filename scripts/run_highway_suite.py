"""Run a predeclared experiment matrix, with fresh project CARLA per episode."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import carla
from analyze_highway_occ import analyze

ROOT=Path(__file__).resolve().parents[1]


def hashes():
    return {f:hashlib.sha256((ROOT/'scripts'/f).read_bytes()).hexdigest() for f in ['highway_occ_core.py','run_highway_occ.py','highway_traffic_worker.py']}


def server(script):
    subprocess.run(['powershell','-NoProfile','-ExecutionPolicy','Bypass','-File',str(ROOT/'scripts'/script)],check=True,cwd=ROOT)


def run(manifest):
    config=json.loads(manifest.read_text());frozen=hashes()
    out=ROOT/'logs/highway-occ'/('suite-'+datetime.now().strftime('%Y%m%d-%H%M%S'))
    out.mkdir(parents=True);(out/'manifest.json').write_text(json.dumps(config,indent=2));(out/'code-hashes.json').write_text(json.dumps(frozen,indent=2))
    results=[]
    print('SUITE',out,flush=True)
    for i,case in enumerate(config['episodes']):
        if hashes()!=frozen:raise RuntimeError('Source changed during frozen matrix; stop rather than mix versions')
        server('Stop-CarlaUE5.ps1');time.sleep(2);server('Start-CarlaUE5.ps1')
        deadline=time.monotonic()+90
        while True:
            try:
                c=carla.Client('127.0.0.1',2100);c.set_timeout(2);c.get_world();break
            except RuntimeError:
                if time.monotonic()>deadline:raise
                time.sleep(2)
        cmd=[sys.executable,'-u',str(ROOT/'scripts/run_highway_occ.py')]
        for k,v in case.items():cmd.extend(['--'+k.replace('_','-'),str(v)])
        print('EPISODE',i+1,case,flush=True)
        process=subprocess.Popen(cmd,cwd=ROOT)
        try:process.wait(timeout=480)
        except subprocess.TimeoutExpired:
            server('Stop-CarlaUE5.ps1')
            try:process.wait(timeout=35)
            except subprocess.TimeoutExpired:process.terminate();process.wait(timeout=10)
        folder=Path(json.loads((ROOT/'logs/highway-occ/current.json').read_text())['folder'])
        if (folder/'report.json').exists():
            result=analyze(folder);(folder/'analysis.json').write_text(json.dumps(result,indent=2));results.append(result)
        else:results.append({'folder':str(folder),'status':'infrastructure_timeout','config':case})
        (out/'results.json').write_text(json.dumps(results,indent=2))
        print('RESULT',json.dumps({k:results[-1].get(k) for k in ['status','seconds','distance_m','max_speed_kmh','collision_events','events','NPC_cutin_events']}),flush=True)
        if (out/'stop.requested').exists():break
    (out/'completion.json').write_text(json.dumps({'episodes_completed':len(results),'expected':len(config['episodes']),'hashes_unchanged':hashes()==frozen},indent=2))
    print('SUITE_DONE',out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('manifest',type=Path);run(p.parse_args().manifest)
