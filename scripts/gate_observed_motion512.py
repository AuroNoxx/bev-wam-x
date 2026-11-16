import json,numpy as np
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'artifacts/carla-observed-motion-paired512'
def main():
 s=json.loads((O/'summary.partial.json').read_text());d=json.loads((R/'data/carla_reactive_hold_cache_v1/manifest.json').read_text());v0=[]
 for n in d['dev']:
  with np.load(n) as z:v0.append(float(z['state'][0]*30))
 per={}
 for arm in s:
  e=json.loads((O/arm/'eval.json').read_text());passed=[];peaks=[]
  for r in e['records']:
   p=np.array(r['points']);vv=p[:,4,None]*p[:,[3,2]];vel=np.vstack(([v0[r['window']],0],vv));a=np.diff(vel,axis=0)/.2;j=np.diff(a,axis=0)/.2;ma=float(np.linalg.norm(a,axis=-1).max());mj=float(np.linalg.norm(j,axis=-1).max());passed.append(ma<=3 and mj<=5);peaks.append([ma,mj])
  per[arm]={'vector_accel_jerk_pass_fraction':float(np.mean(passed)),'vector_peak_accel_mean':float(np.mean(peaks,axis=0)[0]),'vector_peak_jerk_mean':float(np.mean(peaks,axis=0)[1]),'candidate_n':len(passed),'t0_velocity_included':True,'pre_t0_acceleration_not_assumed':True}
 pairs={}
 for seed in [2070,2071]:
  if f'{seed}-observed' not in s or f'{seed}-base' not in s:continue
  b=s[f'{seed}-base']['groups'];o=s[f'{seed}-observed']['groups'];gain=1-o['decelerating']['speed_MAE_0.2_0.6_1_5s'][-1]/b['decelerating']['speed_MAE_0.2_0.6_1_5s'][-1];worst=max(o[g]['speed_MAE_0.2_0.6_1_5s'][i]/b[g]['speed_MAE_0.2_0.6_1_5s'][i]-1 for g in ['command_hold','static_permitted','accelerating','steady'] for i in [1,3]);feasible=per[f'{seed}-observed']['vector_accel_jerk_pass_fraction'];pairs[str(seed)]={'deceleration5s_relative_improvement':gain,'worst_other_group_speed_relative_worsening':worst,'candidate_feasible_fraction':feasible,'passes_all':gain>=.2 and worst<=.1 and feasible>=.95}
 (O/'gates.json').write_text(json.dumps({'arms':per,'pairs':pairs,'closed_loop_allowed':len(pairs)==2 and all(x['passes_all'] for x in pairs.values()),'not_safety_certification':True},indent=2));print(json.dumps({'arms':per,'pairs':pairs},indent=2))
if __name__=='__main__':main()
