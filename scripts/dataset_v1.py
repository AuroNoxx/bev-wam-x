"""Pinned public data, map-disjoint splits, OCC-only current-frame cache."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote
import argparse, gzip, hashlib, json, os, subprocess, tarfile, time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PUBLIC=ROOT/'data/public/bench2drive'
OUT=ROOT/'data/occ_v1'
CFG=json.loads((ROOT/'configs/train_v1.json').read_text())
REV='d84ae18cf20a88b9220df1bc15c073d8824b7ac1'
PREPROCESS_VERSION=2
REPO='rethinklab/Bench2Drive-V0.0.4'
FAMILIES=['HighwayCutIn','HighwayExit','MergerIntoSlowTraffic','MergerIntoSlowTrafficV2','SequentialLaneChange','HardBreakRoute']

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def plan():
    info=json.loads((ROOT/'data/public/bench2drive-file-list.json').read_text())
    if info['sha']!=REV:raise ValueError('Unexpected source revision')
    rows=[]
    for index,family in enumerate(FAMILIES):
        names=[s['rfilename'] for s in info['siblings'] if s['rfilename'].startswith(family+'_') and s['rfilename'].endswith('.tar.gz')]
        names.sort(key=lambda n:hashlib.sha256(n.encode()).hexdigest())
        groups={'train':[n for n in names if '_Town12_' in n],
                'validation':[n for n in names if '_Town13_' in n],
                'test':[n for n in names if '_Town12_' not in n and '_Town13_' not in n]}
        # Reuse already downloaded Town12 data without ever loading old model weights.
        groups['train'].sort(key=lambda n:not (PUBLIC/'archives'/n).exists())
        val_count=[2,1,2,1,1,2][index]
        sizes={'train':7,'validation':val_count,'test':3-val_count}
        for split,count in sizes.items():
            if len(groups[split])<count:raise ValueError((family,split,count))
            for name in groups[split][:count]:rows.append({'file':name,'route':name[:-7],'family':family,'split':split,'revision':REV})
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'plan.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    return rows

def download(row):
    target=PUBLIC/'archives'/row['file'];target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():
        part=target.with_name(target.name+'.part')
        for host in ['https://hf-mirror.com','https://huggingface.co']:
            url=f'{host}/datasets/{REPO}/resolve/{REV}/{quote(row["file"])}'
            r=subprocess.run(['curl.exe','--fail','-L','--retry','2','--retry-delay','2','--connect-timeout','20','--max-time','1200','--speed-limit','1024','--speed-time','60','-sS','-C','-','-o',str(part),url],capture_output=True,text=True)
            if r.returncode==0:part.replace(target);break
        else:raise RuntimeError(f'{target.name}: {r.stderr[-400:]}')
    destination=PUBLIC/'clips'/row['route']
    if not (destination/'.occ_v1_extracted').exists():
        destination.mkdir(parents=True,exist_ok=True)
        # Extract only annotations/OCC; consume the full archive to verify gzip integrity.
        with tarfile.open(target,'r|gz') as tar:
            for member in tar:
                parts=Path(member.name).parts
                kind='anno' if 'anno' in parts else 'occ' if 'occ' in parts else None
                if not kind or not member.isfile():continue
                name=parts[-1]
                if kind=='anno' and not name.endswith('.json.gz'):continue
                if kind=='occ' and not name.endswith('.npz'):continue
                out=destination/kind/name;out.parent.mkdir(exist_ok=True)
                if not out.exists():
                    with tar.extractfile(member) as src,out.open('wb') as dst:
                        while chunk:=src.read(1024*1024):dst.write(chunk)
        (destination/'.occ_v1_extracted').write_text(REV)
    result={**row,'bytes':target.stat().st_size,'sha256':digest(target)}
    (destination/'source_v1.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('DOWNLOADED '+json.dumps(result),flush=True)
    return result

def compact(occ,seg):
    if occ.shape!=(128,128,20):raise ValueError(f'Unexpected source OCC {occ.shape}')
    labels=np.zeros(occ.shape,np.uint8);labels[occ==2]=255;hit=occ==1
    labels[hit]=3
    for cls,ids in [(1,[1,25,27]),(2,[9,10]),(4,[14,15,16,17,18,19,21]),(5,[12,13]),(6,[24]),(7,[2,28])]:
        labels[hit & np.isin(seg,ids)]=cls
    blocks=labels.reshape(64,2,64,2,20)
    pooled=np.zeros((64,64,20),np.uint8)
    pooled[(blocks==255).any(axis=(1,3))]=255
    for cls in [1,2,3,6,7,4,5]:pooled[(blocks==cls).any(axis=(1,3))]=cls
    return pooled.transpose(2,0,1)

def anchors():
    z,y,x=np.indices(CFG['grid']['shape_zyx'])
    # Target is FLU; transform through CARLA x-forward/y-right/z-up.
    return np.stack([-25.6+(x+.5)*.8,25.6-(y+.5)*.8,-5+(z+.5)*.4,np.ones_like(x)],-1).reshape(-1,4)

ANCHORS=anchors()
def warp(grid,world2source,current2world):
    p=ANCHORS @ (world2source @ current2world).T
    x=np.floor((p[:,0]+25.6)/.8).astype(np.int32)
    y=np.floor((p[:,1]+25.6)/.8).astype(np.int32)
    z=np.floor((p[:,2]+5)/.4).astype(np.int32)
    valid=(x>=0)&(x<64)&(y>=0)&(y<64)&(z>=0)&(z<20)
    out=np.full(len(x),255,np.uint8);out[valid]=grid[z[valid],y[valid],x[valid]]
    return out.reshape(CFG['grid']['shape_zyx'])

def prepare_route(row):
    clip=PUBLIC/'clips'/row['route']
    annofiles={int(p.name.split('.')[0]):p for p in (clip/'anno').glob('*.json.gz')}
    occfiles={int(p.stem):p for p in (clip/'occ').glob('*.npz')}
    ids=sorted(annofiles.keys()&occfiles.keys())
    annos={i:json.load(gzip.open(annofiles[i],'rt')) for i in ids}
    poses={i:np.asarray(next(b for b in a['bounding_boxes'] if b['class']=='ego_vehicle')['world2ego'],np.float64) for i,a in annos.items()}
    grids={}
    for i in ids:
        with np.load(occfiles[i],allow_pickle=False) as a:grids[i]=compact(a['occ'],a['seg'])
    target=OUT/'cache'/row['split']/row['route'];target.mkdir(parents=True,exist_ok=True)
    samples=[];bad=[];coverage=[]
    source_sha=json.loads((clip/'source_v1.json').read_text())['sha256'] if (clip/'source_v1.json').exists() else digest(PUBLIC/'archives'/row['file'])
    for i in ids:
        if not all(j in annos for j in range(i-6,i+51)):continue
        # Ego teleports are detected from actual poses, not actor-removal flags for unrelated traffic.
        world_positions=np.stack([np.linalg.inv(poses[j])[:3,3] for j in range(i-6,i+51)])
        if np.linalg.norm(np.diff(world_positions,axis=0),axis=1).max()>8:
            bad.append({'frame':i,'reason':'ego_motion_over_8m_per_0.1s'});continue
        file=target/f'{i:05d}.npz'
        fresh=False
        if file.exists():
            with np.load(file,allow_pickle=False) as cached:fresh='preprocess_version' in cached and int(cached['preprocess_version'])==PREPROCESS_VERSION
        if not fresh:
            ego2world=np.linalg.inv(poses[i]);a=annos[i]
            history=np.stack([warp(grids[i+h],poses[i+h],ego2world) for h in CFG['history_offsets']])
            future=np.stack([warp(grids[i+h],poses[i+h],ego2world) for h in CFG['future_occ_offsets']])
            traj=[]
            for h in range(2,51,2):
                rel=poses[i] @ np.linalg.inv(poses[i+h]);yaw=-np.arctan2(rel[1,0],rel[0,0])
                traj.append([rel[0,3]/50,-rel[1,3]/10,np.sin(yaw),np.cos(yaw),annos[i+h]['speed']/30])
            # Annotation gyro is rad/s (cross-checked against pose derivatives), not CARLA Actor deg/s.
            state=[a['speed']/30,a['acceleration'][0]/10,-a['acceleration'][1]/10,-a['angular_velocity'][2]]
            for which in ['near','far']:
                goal=poses[i]@np.array([a['x_command_'+which],a['y_command_'+which],ego2world[2,3],1])
                state.extend([goal[0]/50,-goal[1]/50])
            for which in ['near','far']:
                command=int(a['command_'+which]);state.extend([float(command==j) for j in range(1,7)])
            tmp=file.with_suffix('.tmp.npz')
            np.savez_compressed(tmp,history=history,future=future,state=np.array(state,np.float32),trajectory=np.array(traj,np.float32),
                                frame=np.int32(i),time_s=np.float64(i*.1),world2ego=poses[i],source_sha=np.array(source_sha),preprocess_version=np.int32(PREPROCESS_VERSION))
            tmp.replace(file)
            coverage.append(float((future!=255).mean()))
        samples.append(str(file.relative_to(OUT)))
    result={**row,'paired_frames':len(ids),'unpaired_annotation_ids':sorted(annofiles.keys()-occfiles.keys()),'unpaired_occ_ids':sorted(occfiles.keys()-annofiles.keys()),
            'windows':len(samples),'samples':samples,'excluded':bad,'max_speed_kmh':max([a['speed']*3.6 for a in annos.values()],default=0),
            'future_coverage_mean_new_cache':float(np.mean(coverage)) if coverage else None,'source_sha256':source_sha}
    print('PREPARED '+json.dumps({k:v for k,v in result.items() if k not in ['samples','excluded']}),flush=True)
    return result

def prepare(rows,pilot=False):
    started=time.monotonic();records=[prepare_route(r) for r in rows]
    splits={s:[p for r in records if r['split']==s for p in r['samples']] for s in ['train','validation','test']}
    report={'version':CFG['version'],'preprocess_version':PREPROCESS_VERSION,'config_sha256':digest(ROOT/'configs/train_v1.json'),'revision':REV,'pilot':pilot,
            'coordinate_frame':'current_ego_FLU_for_history_and_all_future','source_unfilled_is_verified_free':False,
            'frame_time_source':'frame index * 0.1s per source collection cadence; source JSON has no timestamp',
            'split_rule':'Town12 train / Town13 validation / other towns test' if not pilot else 'separate existing routes for cost probe only',
            'routes':records,'splits':splits,'counts':{s:len(v) for s,v in splits.items()},'preprocess_seconds':time.monotonic()-started}
    name='pilot.json' if pilot else 'manifest.json'
    temp=OUT/(name+'.tmp');temp.write_text(json.dumps(report,indent=2),encoding='utf-8');temp.replace(OUT/name)
    print('DATASET '+json.dumps({'manifest':name,'counts':report['counts'],'seconds':report['preprocess_seconds']}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--download',action='store_true');p.add_argument('--prepare',action='store_true');p.add_argument('--pilot',action='store_true');a=p.parse_args()
    rows=plan()
    if a.pilot:
        names=sorted((PUBLIC/'archives').glob('HighwayCutIn*.tar.gz'))[:2]
        rows=[{'file':n.name,'route':n.name[:-7],'family':'HighwayCutIn','split':'train' if i==0 else 'validation','revision':REV} for i,n in enumerate(names)]
        prepare(rows,True)
    else:
        if a.download:
            with ThreadPoolExecutor(max_workers=6) as pool:downloaded=list(pool.map(download,rows))
            (OUT/'downloads.json').write_text(json.dumps(downloaded,indent=2),encoding='utf-8')
        if a.prepare:prepare(rows)
