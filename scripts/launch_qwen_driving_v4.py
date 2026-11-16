from pathlib import Path
import json,time,subprocess
R=Path(__file__).resolve().parents[1];O=R/'artifacts/qwen-driving-v4';O.mkdir(exist_ok=True)
for _ in range(120):
 if (O/'gpu_ready.flag').exists():break
 time.sleep(5)
else:raise TimeoutError('GPU handoff not ready')
m=json.loads((R/'data/qwen_v4_new/manifest.json').read_text());assert len(m['routes'])==4 and m['counts']['train']==789
for name in ['cache_qwen_driving_v4.py','train_qwen_driving_v4.py']:
 with (O/(name+'.log')).open('w',encoding='utf8') as log:
  p=subprocess.Popen([str(R/'.venv-train/Scripts/python.exe'),'-u',str(R/'scripts'/name)],cwd=R,stdout=log,stderr=subprocess.STDOUT);(O/'active_process.json').write_text(json.dumps({'script':name,'pid':p.pid,'started':time.time()}));code=p.wait()
 if code:raise RuntimeError(name+' failed; read log')
