"""Summarize paired logs: predicted vs later executed motion is NOT expert accuracy."""
import json,math
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];base=R/'artifacts/carla-v4-version-pair/tesla'
def main():
 result={'runs':[],'pair_checks':[],'caution':'Collision entries are sensor events, not independent crashes. Motion agreement compares prediction with later executed replanning rollout, not expert trajectory accuracy. Single scenario per condition.'};reports={}
 for version in ('0.9.15','1.0'):
  for scenario in ('follow','brake'):
   path=base/version/(scenario+'-0-model')
   if not (path/'report.json').exists():raise RuntimeError(f'Incomplete run: {path}')
   rep=json.loads((path/'report.json').read_text());rows=json.loads((path/'steps.json').read_text());reports[version,scenario]=rep
   ts=np.array([r['timestamp'] for r in rows]);xy=np.array([r['ego_location'][:2] for r in rows]);errs=[]
   for row in rows:
    target_t=row['timestamp']+np.arange(1,26)*.2
    if target_t[-1]>ts[-1]+1e-6:continue
    future=np.stack([np.interp(target_t,ts,xy[:,i]) for i in (0,1)],1);delta=future-np.array(row['ego_location'][:2]);yaw=math.radians(row['ego_yaw_deg']);local=np.stack([delta[:,0]*math.cos(yaw)+delta[:,1]*math.sin(yaw),delta[:,0]*math.sin(yaw)-delta[:,1]*math.cos(yaw)],1)
    errs.append(np.linalg.norm(np.array(row['output']['xy'])-local,axis=1))
   errors=np.array(errs);lane=np.array([r['nearest_lane_distance_m'] for r in rows]);item=dict(version=version,scenario=scenario,status=rep.get('status'),path_length_logged_m=float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum()),displacement_logged_m=float(np.linalg.norm(xy[-1]-xy[0])),reported_displacement_m=rep.get('displacement_m'),lane_mean_m=float(lane.mean()),lane_max_m=float(lane.max()),collision_sensor_events=len(rep.get('collision_events',[])),collision_actor_types=sorted(set(x['other_actor'] for x in rep.get('collision_events',[]))),agreement_windows=len(errs),prediction_executed_ADE_m=float(errors.mean()) if len(errs) else None,prediction_executed_FDE5s_m=float(errors[:,-1].mean()) if len(errs) else None)
   result['runs'].append(item)
 for scenario in ('follow','brake'):
  a,b=reports['0.9.15',scenario],reports['1.0',scenario];ar=np.asarray(a['route_xyz']);br=np.asarray(b['route_xyz']);result['pair_checks'].append(dict(scenario=scenario,spawn_equal=a['spawn_transform']==b['spawn_transform'],route_shape_equal=ar.shape==br.shape,route_max_abs_diff_m=float(abs(ar-br).max()) if ar.shape==br.shape else None,xodr_sha_equal=a['xodr_sha256']==b['xodr_sha256'],weather_equal=a.get('weather')==b.get('weather'),vehicle_equal=a.get('vehicle_type')==b.get('vehicle_type'),camera_equal=a.get('camera')==b.get('camera')))
 for check in result['pair_checks']:
  a,b=reports['0.9.15',check['scenario']],reports['1.0',check['scenario']]
  for key in ('vehicle_mass','bbox_extent','wheel_max_steer'):
   av,bv=a.get(key),b.get(key)
   check[key+'_0915']=av;check[key+'_1_0']=bv
   check[key+'_equal']=av is not None and bv is not None and av==bv
   check[key+'_delta_1_0_minus_0915']=(np.asarray(bv)-np.asarray(av)).tolist() if av is not None and bv is not None else None
 try:
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  fig,axes=plt.subplots(1,2,figsize=(12,7))
  for ax,scenario in zip(axes,('follow','brake')):
   plotted=[]
   route=np.asarray(reports['0.9.15',scenario]['route_xyz']);ax.plot(route[:,0],route[:,1],'k--',alpha=.35,label='Provided route')
   for version,color in (('0.9.15','tab:blue'),('1.0','tab:orange')):
    rows=json.loads((base/version/(scenario+'-0-model')/'steps.json').read_text());xy=np.asarray([r['ego_location'][:2] for r in rows]);plotted.append(xy);ax.plot(xy[:,0],xy[:,1],color=color,label=version+' executed');ax.scatter(xy[0,0],xy[0,1],color=color,marker='o');ax.scatter(xy[-1,0],xy[-1,1],color=color,marker='x')
   ax.set_title(scenario+' | Tesla Model 3');ax.set_xlabel('World X (m)');ax.set_ylabel('World Y (m)');union=np.concatenate(plotted);ax.set_xlim(union[:,0].min()-5,union[:,0].max()+5);ax.set_ylim(union[:,1].min()-5,union[:,1].max()+5);ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.2);ax.legend()
  fig.suptitle('Frozen Qwen-v4 | 20s / single rollout per condition (not benchmark scores)');fig.tight_layout();fig.savefig(base/'trajectories.png',dpi=180);plt.close(fig);result['trajectory_png']='trajectories.png'
 except ImportError:result['trajectory_png']=None
 (base/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
