"""Read only selected training cache audit; no test data or external service."""
import json,numpy as np
from pathlib import Path
root=Path(__file__).resolve().parents[1]
s=json.loads((root/'artifacts/joint-occ-v1/audit/selection.json').read_text())
rows=[]
for name in s['splits']['train']:
    with np.load(root/'data/occ_v3'/name) as f:
        h=f['history'];future=f['future'][:2];support=(h[-1]>0)&(h[-1]<8)
        occupied=(future>0)&(future<8);vehicle=future==4
        rows.append({'file':name,'future_occupied':int(occupied.sum()),'retained_occupied':int((occupied&support).sum()),'future_vehicle':int(vehicle.sum()),'retained_vehicle':int((vehicle&support).sum()),'state_finite':bool(np.isfinite(f['state']).all()),'trajectory_finite':bool(np.isfinite(f['trajectory']).all()),'history_occ_pose_offset_minus1':bool(np.array_equal(f['history_occ_ids'],f['history_pose_ids']-1))})
result={'windows':len(rows),'scope':'selected training only; counts aggregated over horizon and voxels; occupied semantics 1..7, vehicle class4 per existing model contract','counts':{k:sum(x[k] for x in rows) for k in ['future_occupied','retained_occupied','future_vehicle','retained_vehicle']},'all_finite':all(x['state_finite'] and x['trajectory_finite'] for x in rows),'all_offsets_correct':all(x['history_occ_pose_offset_minus1'] for x in rows),'test_read':False,'source_causality_certified':False,'rows':rows}
for k in ['occupied','vehicle']:
    result[k+'_retained_fraction']=result['counts']['retained_'+k]/max(1,result['counts']['future_'+k])
p=root/'artifacts/joint-occ-training-support-audit.json'
assert not p.exists();p.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
