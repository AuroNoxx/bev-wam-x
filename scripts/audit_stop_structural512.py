"""Frozen post-training stop/deceleration diagnostic; train split only, no updates."""
from train_structural_prototypes_light import *

def main():
 idle();configure_runtime();base=ROOT/'artifacts/structural-prototypes-light512';assert (base/'summary.json').exists();out=ROOT/'artifacts/implementation-plan-v2/stop-frozen512';out.mkdir(exist_ok=False)
 select=json.loads((out.parent/'stop_deceleration_train_only_selection.json').read_text());names=[r['file'] for r in select['selection']];ds=DiskSamples(names);labels=[r['trajectory'].numpy()*np.array([50,10,1,1,30]) for r in ds];cfg=json.loads((base/'config.json').read_text());scales=json.loads((ROOT/'artifacts/residual-representation-paired128/scale_calibration.json').read_text())['scales_m'];results=[]
 for seed in [2010,2011]:
  for arm in ['baseline','velocity_only','world_only']:
   idle();src=base/f'{seed}-{arm}'/'latest.ckpt';ck=torch.load(src,map_location='cpu',weights_only=False);assert ck['global_step']==512;task=ProtoTask(cfg,seed,arm,scales,'frozen-only');task.load_state_dict(ck['state_dict'],strict=True);del ck;task.model.cuda().eval();reference=copy.deepcopy(task.model.teacher);ev=evaluate(task.model,names,reference,True);groups={}
   for kind in ['initial_stopped_and_end_stopped','moving_to_stopped','decelerating_not_stopped']:
    rows=[]
    for r in ev['records']:
     i=r['window'];info=select['selection'][i]
     if info['stratum']!=kind:continue
     p=np.array(r['points']);truth=labels[i];v=p[:,4:5]*p[:,[3,2]];acc=np.diff(np.vstack(([info['initial_speed_mps'],0.],v)),axis=0)/.2;jerk=np.diff(acc,axis=0)/.2;rows.append({**{k:r[k] for k in ['kinematic_mps','first_residual_mps','max_accel_mps2','max_jerk_mps3','forward_proxy_m']},'vector_peak_accel_mps2':float(np.linalg.norm(acc,axis=1).max()),'vector_peak_jerk_mps3':float(np.linalg.norm(jerk,axis=1).max()),'predicted_end_stop':float(p[-1,4]<.5),'final_speed_mps':float(p[-1,4]),'initial_speed_mps':info['initial_speed_mps'],'label_final_speed_mps':info['label_final_speed_mps'],'path_length_m':float(np.linalg.norm(np.diff(np.vstack(([0,0],p[:,:2])),axis=0),axis=1).sum()),'position_label_fit_m':float(np.linalg.norm(p[:,:2]-truth[:,:2],axis=1).mean())})
    groups[kind]={'windows':len(rows)//4,'candidates':len(rows),'mean':{k:float(np.mean([r[k] for r in rows])) for k in rows[0]} if rows else None}
   r={'seed':seed,'arm':arm,'checkpoint':str(src.relative_to(ROOT)),'sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'groups':groups,'raw':ev};(out/f'{seed}-{arm}.json').write_text(json.dumps(r,indent=2));results.append({k:v for k,v in r.items() if k!='raw'});print(json.dumps(results[-1]),flush=True);del task,reference;gc.collect();torch.cuda.empty_cache()
 (out/'summary.json').write_text(json.dumps({'results':results,'limits':select['limits'],'collision':None,'TTC':None,'road_validity':None,'parking_acceptance':False,'four_candidates_not_independent':True,'actual_closed_loop':False},indent=2))
if __name__=='__main__':main()

