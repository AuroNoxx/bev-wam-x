from pathlib import Path
import subprocess,time,socket,json,psutil
R=Path(__file__).resolve().parents[1];O=R/'artifacts/qwen-driving-v4';server=None
for _ in range(360):
 try:
  if json.loads((O/'completed.json').read_text()).get('epochs')==20:break
 except (OSError,ValueError):pass
 time.sleep(5)
else:raise TimeoutError('v4 training not complete')
try:
 exe=R/'runtime/CARLA-1.0/Carla-1.0-Win64-Shipping/CarlaUnreal.exe';si=subprocess.STARTUPINFO();si.dwFlags|=subprocess.STARTF_USESHOWWINDOW;si.wShowWindow=0;server=subprocess.Popen([str(exe),'-RenderOffScreen','-quality-level=Low','-dx12','-carla-rpc-port=2102','-nosound','-ResX=448','-ResY=252'],startupinfo=si)
 for _ in range(90):
  try:s=socket.create_connection(('localhost',2102),timeout=1);s.close();break
  except OSError:time.sleep(1)
 else:raise TimeoutError('CARLA port not ready')
 for seed in [0,1]:
  for scenario in ['follow','brake','lanechange']:
   for mode in ['model','baseline']:
    progress={'phase':'scenarios','seed':seed,'scenario':scenario,'mode':mode};(O/'scenario_progress.json').write_text(json.dumps(progress))
    with (O/f'scenario-{scenario}-{seed}-{mode}.log').open('w') as log:
     p=subprocess.run([str(R/'.venv-carla-ue5/Scripts/python.exe'),str(R/'scripts/run_carla_scenario_v4.py'),mode,scenario,str(seed)],cwd=R,stdout=log,stderr=subprocess.STDOUT,timeout=180)
    if p.returncode:raise RuntimeError('Scenario failed '+str(progress))
 (O/'scenarios_completed.json').write_text(json.dumps({'completed_runs':12,'scenarios':['follow','brake','lanechange'],'seeds':[0,1]}))
except Exception as e:
 (O/'scenario_error.json').write_text(json.dumps({'error':repr(e)}));raise
finally:
 stopped=[];root=(R/'runtime/CARLA-1.0').resolve()
 for p in psutil.process_iter(['pid','exe','cmdline']):
  try:
   if p.info['exe'] and root in Path(p.info['exe']).resolve().parents and '-carla-rpc-port=2102' in (p.info['cmdline'] or []):p.terminate();stopped.append(p.pid)
  except psutil.NoSuchProcess:pass
 (O/'scenario_cleanup.json').write_text(json.dumps({'stopped_own_port2102':stopped}))
