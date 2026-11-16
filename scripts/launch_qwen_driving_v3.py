import json,time,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'artifacts/qwen-driving-v3'
for i in range(720):
 p=O/'live_progress.json'
 try:
  d=json.loads(p.read_text())
  if d.get('phase')=='caching' and d.get('completed')==6420:break
 except (OSError,ValueError):pass
 time.sleep(5)
else:raise TimeoutError('Cache not complete after1h')
with (O/'training_console.log').open('w',encoding='utf8') as log:
 p=subprocess.Popen([str(R/'.venv-train/Scripts/python.exe'),'-u',str(R/'scripts/train_qwen_driving_v3.py')],cwd=R,stdout=log,stderr=subprocess.STDOUT)
 (O/'training_process.json').write_text(json.dumps(dict(pid=p.pid,started_at=time.time())));code=p.wait()
if code:raise RuntimeError('Training failed; read training_console.log')
