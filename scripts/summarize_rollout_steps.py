import json
from pathlib import Path
root=Path(__file__).resolve().parents[1];out=root/'artifacts/teacher-rollout-steps-paired';rows=[]
for seed in [1810,1811]:
 for mode in ['rollout4','rollout16']:
  r=json.loads((out/f'{seed}-{mode}/report.json').read_text());s=r['after']['summary'];a=r['after'];g=r['gradient_log'];rows.append({'seed':seed,'mode':mode,'steps':r['steps'],'train_wall_s':r['wall_seconds'],'evaluation_wall_s':r['evaluation_seconds'],'allocated_gib':r['peak_allocated_gib'],'reserved_gib':r['peak_reserved_gib'],**s,'global_first_accel':a['peak_first_abs_accel_mps2'],'global_accel':max(x['max_accel_mps2'] for x in a['records']),'global_jerk':max(x['max_jerk_mps3'] for x in a['records']),'clipped_steps':sum(x['clipped'] for x in g),'world_diagnostic':a['separate_2window_diagnostics']})
result={'rows':rows,'measured_train_wall_sum_s':sum(r['train_wall_s'] for r in rows),'measured_evaluation_wall_sum_s':sum(r['evaluation_wall_s'] for r in rows),'limitations':'2seeds x2arms x64updates, 128train16DEVELOPMENT windows; source OCC causality uncertified, no collision/TTC/closed-loop safety. Per-trajectory peak averages distinguished from global maxima. World diagnostic2windows only.'};(out/'comparison.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
