"""Frozen offline input ablations from real closed-loop observations, never actuates."""
import argparse
import json
from pathlib import Path
import queue
import subprocess
import threading
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    runs=Path(a.runs);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    responses=queue.Queue();stderr=(out/'worker.stderr.log').open('w')
    worker=subprocess.Popen([str(ROOT/'.venv-train/Scripts/python.exe'),'-u',str(ROOT/'scripts/native_v2_carla_worker.py'),'--output',str(out.resolve())],
                            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=stderr,text=True,cwd=ROOT)
    def read():
        for line in worker.stdout:responses.put(line)
    threading.Thread(target=read,daemon=True).start();ready=json.loads(responses.get(timeout=120));assert ready.get('ready')
    results=[]
    try:
        for scene,when in [('turn',13.),('turn',15.),('obstacle',10.),('obstacle',20.),('dynamic',11.05),('dynamic',11.85),('dynamic',15.05)]:
            folder=runs/scene;rows=[json.loads(s) for s in (folder/'plans.jsonl').read_text().splitlines()]
            row=min(rows,key=lambda r:abs(r['sim_t']-when));stem=scene+'-'+str(row['id'])
            with np.load(folder/row['input_file']) as z:history=z['history'].copy();state=z['state'].copy()
            variants={'recorded':(history,state)}
            no_dynamic=history.copy();z,y,x=np.indices(history.shape[1:]);xx=-25.6+(x+.5)*.4;yy=25.6-(y+.5)*.4
            report=json.loads((folder/'report.json').read_text());extent=report['ego_bounding_box']['extent']
            front=(xx>0)&~((np.abs(xx)<extent[0]+.4)&(np.abs(yy)<extent[1]+.4))
            remove=np.isin(no_dynamic,[4,5])&front[None]
            no_dynamic[remove]=0  # unfilled, not asserted verified free
            variants['remove_dynamic_ahead_diagnostic_only']=(no_dynamic,state)
            no_navigation=state.copy();no_navigation[4:]=0
            variants['zero_navigation_diagnostic_only']=(history,no_navigation)
            predictions={}
            for name,(h,s) in variants.items():
                file=out/(stem+'-'+name+'.npz');np.savez_compressed(file,history=h,state=s)
                worker.stdin.write(json.dumps({'input':str(file.resolve()),'seed':row['prediction']['seed']})+'\n');worker.stdin.flush()
                prediction=json.loads(responses.get(timeout=10));assert 'error' not in prediction,prediction
                predictions[name]=prediction
            base=np.asarray(predictions['recorded']['xy_flu_m']);original=np.asarray(row['prediction']['xy_flu_m'])
            result={'scene':scene,'plan_id':row['id'],'sim_t':row['sim_t'],'seed':row['prediction']['seed'],
                    'input_file':row['input_file'],'replay_max_xy_difference_m':float(np.abs(base-original).max()),
                    'removed_dynamic_voxels_ahead':int(remove.sum()),'predictions':predictions,'effects':{}}
            for name in variants:
                xy=np.asarray(predictions[name]['xy_flu_m']);vel=np.asarray(predictions[name]['velocity_flu_mps'])
                result['effects'][name]={'first_forward_velocity_mps':float(vel[0,0]),'endpoint5s_xy_m':xy[-1].tolist(),
                                         'mean_xy_difference_from_recorded_m':float(np.linalg.norm(xy-base,axis=1).mean()),
                                         'endpoint5s_difference_m':float(np.linalg.norm(xy[-1]-base[-1]))}
            results.append(result);(out/'results.json').write_text(json.dumps({'results':results,
                'scope':'Frozen offline artificial input ablations, identical action noise, no actuation; no causal claim about an unobserved real traffic intervention',
                'weights_modified':False,'training_steps':0},indent=2))
            print(json.dumps({k:result[k] for k in ['scene','sim_t','replay_max_xy_difference_m','removed_dynamic_voxels_ahead','effects']}),flush=True)
    finally:
        worker.stdin.write('{"close":true}\n');worker.stdin.flush();worker.wait(timeout=15);stderr.close()


if __name__=='__main__':main()
