from pathlib import Path
import json,time,subprocess,hashlib
R=Path(__file__).resolve().parents[1];L=R/'logs/bench2drive-0915';D=R/'runtime/CARLA-0.9.15';seven=Path('C:/Program Files/7-Zip/7z.exe')
def state(phase,**kw):
 (L/'install-status.json').write_text(json.dumps(dict(phase=phase,time=time.time(),**kw),indent=2));print(phase,kw,flush=True)
try:
 for name,cache,size in [('CARLA_0.9.15.zip','carla-0915',7755135185)]:
  p=R/'.cache'/cache/name;state('waiting_for_archive',archive=str(p))
  while not p.exists() or p.stat().st_size!=size:time.sleep(20)
  state('integrity_check',archive=str(p))
  with (L/(name+'.verify.log')).open('w') as log:subprocess.run([str(seven),'t',str(p),'-bsp0'],stdout=log,stderr=subprocess.STDOUT,check=True)
  state('extracting',archive=str(p));D.mkdir(exist_ok=True)
  with (L/(name+'.extract.log')).open('w') as log:subprocess.run([str(seven),'x',str(p),'-o'+str(D),'-y','-bsp0'],stdout=log,stderr=subprocess.STDOUT,check=True)
 state('installed_requires_smoke',executables=[str(p) for p in D.rglob('CarlaUE4.exe')],note='Do not start full benchmark until smoke/interface checks pass')
except Exception as e:state('failed',error=repr(e));raise

