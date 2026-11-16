import json,numpy as np
from pathlib import Path
b=Path('artifacts/carla-identifiability-frozen');d=json.load(open(b/'data.json'));r=json.load(open(b/'model.json'));out={}
for split in ['train','dev']:
 s=[x for x in d if x['split']==split]; moving=[x for x in s if min(x['speed'])>.5 and max(abs(y) for y in x['y06'])>.005];out[split]={'pairs':len(s),'moving_lateral_pairs':len(moving),'label_opposite_sign_fraction':float(np.mean([x['y06'][0]*x['y06'][1]<0 for x in moving])),'label_signed_sum_abs_mean':float(np.mean([abs(sum(x['y06'])) for x in moving])),'instant_nav_matches_y06_fraction':float(np.mean([n*y>0 for x in moving for n,y in zip(x['nav_y'],x['y06'])])),'occ_union_diff_mean':float(np.mean([x['occ_union_difference_fraction'] for x in moving]))}
 print(split,out[split])
 for x in r:
  if x['meta']['split']!=split:continue
  o=x['outputs'];print(x['meta']['idx'],'v',round(x['meta']['speed'][0],2),'target',np.round(x['meta']['y06'],4),'pred',np.round(o['actual']['y06'],4),'navdelta',np.round(np.array(o['nav_swap']['y06'])-o['actual']['y06'],5),'occdelta',np.round(np.array(o['occ_swap']['y06'])-o['actual']['y06'],5),'grad',[round(g['single_flow_vy_derivative_per_nav_meter'],6) for g in x['single_step_fp32_gradients']])
(b/'summary.json').write_text(json.dumps(out,indent=2))
