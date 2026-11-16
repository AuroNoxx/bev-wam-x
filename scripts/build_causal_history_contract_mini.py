"""New small causal-history data contract; source causality remains uncertified."""
import json,gzip,hashlib
from pathlib import Path
import numpy as np
from dataset_v1 import ROOT,compact,warp
from probe_teacher_history_matrix import idle

def main():
    idle();out=ROOT/'artifacts/causal-history-contract-mini';out.mkdir(exist_ok=False)
    sel=json.loads((ROOT/'artifacts/joint-occ-v1/audit/selection.json').read_text());manifest=json.loads((ROOT/'data/occ_v3/manifest.json').read_text())
    names=sel['splits']['train'][:8]+[r['samples'][0] for r in manifest['routes'] if r['split']=='train'][:8];names=list(dict.fromkeys(names));offsets=[-50,-40,-30]+list(range(-20,1,2));results=[]
    for index,name in enumerate(names):
        i=int(Path(name).stem);route=Path(name).parts[2];clip=ROOT/'data/public/bench2drive/clips'/route
        annos={};poses={};available={int(p.name.split('.')[0]):p for p in (clip/'anno').glob('*.json.gz')}
        for j in range(max(0,i-50),i+1):
            if j not in available:continue
            with gzip.open(available[j],'rt') as f:a=json.load(f)
            pose=np.asarray(next(b['world2ego'] for b in a['bounding_boxes'] if b['class']=='ego_vehicle'),float);assert np.isfinite(pose).all();annos[j]=a;poses[j]=pose
        boundary=i;discontinuities=[];maxjump=0.
        for j in range(i,i-50,-1):
            if j not in poses or j-1 not in poses:discontinuities.append({'frame':j,'cause':'missing_annotation_or_start'});break
            jump=float(np.linalg.norm(np.linalg.inv(poses[j])[:3,3]-np.linalg.inv(poses[j-1])[:3,3]));maxjump=max(maxjump,jump)
            if jump>=8:discontinuities.append({'frame':j,'cause':'ego_jump_over8m'});break
            if not (clip/'occ'/f'{j-2:05d}.npz').exists():discontinuities.append({'frame':j-1,'cause':'missing_paired_occ'});break
            boundary=j-1
        histories=[];geometries=[];observed=[];valid=[]
        for offset in offsets:
            j=i+offset;ok=j>=boundary and j in poses and (clip/'occ'/f'{j-1:05d}.npz').exists();valid.append(ok)
            if ok:
                with np.load(clip/'occ'/f'{j-1:05d}.npz') as f:h=warp(compact(f['occ'],f['seg']),poses[j],np.linalg.inv(poses[i]))
                g=warp(np.ones((20,64,64),np.uint8),poses[j],np.linalg.inv(poses[i]))==1
            else:h=np.full((20,64,160),255,np.uint8);g=np.zeros_like(h,dtype=bool)
            histories.append(h);geometries.append(g);observed.append((h>0)&(h<8))
        history=np.stack(histories);valid=np.array(valid);geometry=np.stack(geometries);occupied=np.stack(observed)
        with np.load(ROOT/'data/occ_v3'/name) as old:
            assert np.array_equal(history[-4:],old['history'])
            state=old['state'].copy();future=old['future'].copy();trajectory=old['trajectory'].copy()
        assert not geometry[~valid].any() and (history[~valid]==255).all()
        actor={'history':history,'state':state,'history_valid':valid,'history_age_seconds':-np.array(offsets,dtype=np.float32)*.1,'geometry_valid':geometry,'occupied_observed':occupied,'history_pose_ids':i+np.array(offsets),'history_occ_ids':i+np.array(offsets)-1,'anchor_world2ego':poses[i]}
        assert all(k not in actor for k in ['future','trajectory','future_mask'])
        ap=out/f'{index:02d}.actor.npz';lp=out/f'{index:02d}.labels.npz';np.savez_compressed(ap,**actor);np.savez_compressed(lp,future=future,trajectory=trajectory)
        # Separate label mutations cannot change actor loader results.
        with np.load(ap) as stored:assert set(stored.files)==set(actor) and all(np.array_equal(stored[k],v) for k,v in actor.items())
        poisoned_future=np.full_like(future,255);assert actor['history'].tobytes()==history.tobytes()
        results.append({'source':name,'actor':ap.name,'labels':lp.name,'valid_history_frames':int(valid.sum()),'missing_history_frames':int((~valid).sum()),'continuous_boundary_pose_id':boundary,'max_observed_ego_step_m':maxjump,'boundary_reasons':discontinuities,'four_frame_match':True,'actor_sha256':hashlib.file_digest(ap.open('rb'),'sha256').hexdigest()})
    result={'windows':len(results),'history_offsets_frames':offsets,'test_read':False,'source_causality_certified':False,'timestamp_source':'nominal0.1s index cadence, not native timestamp','rules':'reset history at missing paired annotation/OCC or >=8m ego step; geometry-valid differs from occupied-observed; source0 never assumed free','results':results,'model_integration':'not enabled: existing actor does not yet consume history_valid/age; no training on padded sequences'}
    (out/'manifest.json').write_text(json.dumps(result,indent=2));print(json.dumps({'windows':len(results),'missing_frames':sum(r['missing_history_frames'] for r in results),'all_four_frame_match':all(r['four_frame_match'] for r in results),'max_ego_step_m':max(r['max_observed_ego_step_m'] for r in results)}))
if __name__=='__main__':main()
