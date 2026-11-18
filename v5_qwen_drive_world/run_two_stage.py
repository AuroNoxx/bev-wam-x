"""Sequential process launcher; Lightning remains the sole training framework."""
import json,os,subprocess,sys,time,traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    status=ROOT/'artifacts/v5-qwen-drive-world/two-stage-run.json'
    python=ROOT/'.venv-train/Scripts/python.exe'
    report={'pid':os.getpid(),'started_unix':time.time(),'stages':[],'metric_gates':False}
    def save():status.write_text(json.dumps(report,indent=2),encoding='utf-8')
    try:
        for phase in ('overfit64','subset2000'):
            folder=ROOT/'artifacts/v5-qwen-drive-world'/phase
            command=[str(python),'-m','v5_qwen_drive_world.train_supervised','--phase',phase]
            if (folder/'latest.ckpt').exists():command+=['--resume']
            elif phase=='subset2000':command+=['--initialize-from',str(ROOT/'artifacts/v5-qwen-drive-world/overfit64/latest.ckpt')]
            report.update(state='running',active_phase=phase);save()
            log=ROOT/f'logs/v5-qwen-drive-world/{phase}-training.log'
            with log.open('a',encoding='utf-8') as stream:
                process=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
                report['child_pid']=process.pid;save();code=process.wait()
            if code:raise RuntimeError(f'{phase} exited {code}; see {log}')
            current=json.loads((folder/'status.json').read_text())
            if current['state']!='finished':raise RuntimeError(f'{phase} did not finish all requested epochs')
            report['stages'].append(current);save()
        report.update(state='completed',finished_unix=time.time(),child_pid=None);save()
    except Exception:
        report.update(state='failed',error=traceback.format_exc(),finished_unix=time.time());save();raise


if __name__=='__main__':main()
