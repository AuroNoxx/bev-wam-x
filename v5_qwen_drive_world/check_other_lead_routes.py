"""Score the fixed readout on eligible routes absent from its fit and development sets."""
import json
from .lead_motion import DEST,RealData,displacement,frame,load_base,observation,torch
from .train_lead_motion import LeadReadout,metrics


@torch.no_grad()
def main():
    folder=DEST/'directed'
    used={r['route'] for r in json.loads((folder/'labels-and-split.json').read_text())['windows']}
    data=RealData('subset2000')
    candidates=[r for r in data.records if r['source']=='bench2drive' and r['speed_kmh']<=3 and r['route'] not in used]
    eligible=[]
    for r in candidates:
        target=displacement(r['route'],r['index'])
        if target is not None:
            eligible.append((r,target))
    rows=[]
    if eligible:
        model=load_base(data)
        x=torch.load(folder/'observations.pt',weights_only=True)
        head=LeadReadout(x,{},[]).cuda().eval()
        ckpt=torch.load(folder/'latest.ckpt',map_location='cpu',weights_only=False)
        assert ckpt['global_step']==700
        head.load_state_dict(ckpt['state_dict'])
        for r,target in eligible:
            cameras=[frame(r['route'],r['index']+offset)['sensors']['CAM_FRONT'] for offset in (-15,-10,-5,0)]
            obs=observation(model,r['actor_path'],cameras,directed=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                pred=float(head(obs[None].cuda()).item())
            rows.append(dict(route=r['route'],index=r['index'],target=target,prediction=pred))
    summary=dict(candidates=len(candidates),candidate_routes=len({r['route'] for r in candidates}),
                 routes=len({r['route'] for r in rows}),metrics=metrics(rows) if rows else None)
    (folder/'other-training-routes.json').write_text(json.dumps(dict(summary=summary,windows=rows),indent=2))
    print(summary,flush=True)


if __name__=='__main__':
    main()
