from pathlib import Path
import json,subprocess,time,socket,psutil
R=Path(__file__).resolve().parents[1];L=R/'logs/bench2drive-0915';O=R/'artifacts/b2d-0915';P=R/'.venv-b2d0915/Scripts/python.exe'
def state(phase,**kw):
 (L/'pipeline-status.json').write_text(json.dumps(dict(phase=phase,time=time.time(),**kw),indent=2));print(phase,kw,flush=True)
state('prepared_not_started',note='Run explicitly after dependencies and archive install checks. No training.')
