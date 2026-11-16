"""Build independent RGB-only v4 state/trajectory labels from B2D annotations.
No OCC required, no old validation/test edits. Runs old-cache parity before any new label writes.
"""
from pathlib import Path
import argparse,gzip,io,json,tarfile,hashlib,time
import numpy as np
R=Path(__file__).resolve().parents[1];O=R/'data/qwen_v4_new'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while b:=f.read(4*1024*1024):h.update(b)
 return h.hexdigest()
def load_clips(clip):
 return {int(p.name.split('.')[0]):json.load(gzip.open(p,'rt')) for p in sorted((clip/'anno').glob('*.json.gz'))}
def load_tar(path):
 out={}
 with tarfile.open(path,'r|gz') as t:
  for z in t:
   if z.isfile() and 'anno' in Path(z.name).parts and z.name.endswith('.json.gz'):
    i=int(Path(z.name).name.split('.')[0]);assert i not in out,'duplicate annotation id';out[i]=json.loads(gzip.decompress(t.extractfile(z).read()))
 return out
def poses_for(ann):
 return {i:np.asarray(next(b for b in a['bounding_boxes'] if b['class']=='ego_vehicle')['world2ego'],np.float64) for i,a in ann.items()}
def construct(ann,poses,i):
 ego2world=np.linalg.inv(poses[i]);a=ann[i];traj=[]
 for h in range(2,51,2):
  rel=poses[i]@np.linalg.inv(poses[i+h]);yaw=-np.arctan2(rel[1,0],rel[0,0]);traj.append([rel[0,3]/50,-rel[1,3]/10,np.sin(yaw),np.cos(yaw),ann[i+h]['speed']/30])
 state=[a['speed']/30,a['acceleration'][0]/10,-a['acceleration'][1]/10,-a['angular_velocity'][2]]
 for which in ['near','far']:
  goal=poses[i]@np.array([a['x_command_'+which],a['y_command_'+which],ego2world[2,3],1]);state.extend([goal[0]/50,-goal[1]/50])
 for which in ['near','far']:
  command=int(a['command_'+which]);state.extend([float(command==j) for j in range(1,7)])
 st=np.array(state,np.float32);tr=np.array(traj,np.float32);assert st.shape==(20,) and tr.shape==(25,5);assert np.isfinite(st).all() and np.isfinite(tr).all()
 return {'state':st,'trajectory':tr,'frame':np.int32(i),'time_s':np.float64(i*.1),'world2ego':poses[i],'history_frame_ids':np.array([i-6,i-4,i-2,i],np.int32),'future_trajectory_frame_ids':np.arange(i+2,i+51,2,dtype=np.int32),'future_label_frame_ids':np.array([i+14,i+16,i+18,i+20],np.int32)}
def parity():
 old=json.loads((R/'data/occ_v3/manifest.json').read_text());reports=[]
 for family in ['HighwayCutIn','HardBreakRoute','SequentialLaneChange']:
  route=next(x for x in old['routes'] if x['family']==family and x['split']=='train');ann=load_clips(R/'data/public/bench2drive/clips'/route['route']);poses=poses_for(ann);name=route['samples'][len(route['samples'])//2];i=int(Path(name).stem);new=construct(ann,poses,i)
  with np.load(R/'data/occ_v3'/name) as z:
   errors={key:float(np.max(abs(new[key]-z[key]))) for key in ['state','trajectory','world2ego','frame','time_s']};assert all(v==0 for v in errors.values()),errors
  reports.append({'route':route['route'],'frame':i,'max_abs_error':errors,'old_sample':name})
 return reports
def atomic_json(path,data):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.pending.json');tmp.write_text(json.dumps(data,indent=2),encoding='utf8');tmp.replace(path)
def main():
 p=argparse.ArgumentParser();p.add_argument('--plan');p.add_argument('--archive-dir',default=str(R/'data/public/bench2drive/archives'));p.add_argument('--clips-dir');p.add_argument('--download-manifest');p.add_argument('--verify-only',action='store_true');args=p.parse_args();proof=parity();O.mkdir(exist_ok=True);atomic_json(O/'parity_report.json',proof);print('PARITY_PASS',json.dumps(proof),flush=True)
 if args.verify_only:return
 assert args.plan,'--plan is required';raw=json.loads(Path(args.plan).read_text(encoding='utf-8-sig'));rows=raw if isinstance(raw,list) else raw.get('routes',raw.get('selected',raw.get('plan')));assert isinstance(rows,list)
 old=json.loads((R/'data/occ_v3/manifest.json').read_text());oldnames={x['route'] for x in old['routes']};records=[];pending=[];downloaded={}
 if args.download_manifest:downloaded={x['route']:x for x in json.loads(Path(args.download_manifest).read_text())['items']}
 for row in rows:
  if isinstance(row,str):row={'file':row}
  name=row.get('file',row.get('filename',row.get('rfilename')));route=row.get('route',name[:-7] if name else None);assert route and '_Town12_' in route and route not in oldnames,'Only new Town12 training routes allowed';archive=Path(args.archive_dir)/(name or route+'.tar.gz')
  if args.download_manifest and downloaded.get(route,{}).get('status')!='complete':pending.append({'route':route,'reason':'download/extraction not verified complete'});continue
  if not archive.exists():pending.append({'route':route,'reason':'archive not yet available'});continue
  try:
   clip=Path(downloaded[route]['annotation_directory']).parent if route in downloaded else (Path(args.clips_dir)/route if args.clips_dir else None);ann=load_clips(clip) if clip and (route in downloaded or (clip/'.annotations_complete').exists()) else load_tar(archive);assert ann;poses=poses_for(ann);ids=sorted(ann);positions={j:np.linalg.inv(poses[j])[:3,3] for j in ids};source=sha(archive)
   if row.get('expected_sha256'):assert source==row['expected_sha256'],'source hash mismatch'
   if route in downloaded:assert len(ann)==downloaded[route]['annotation_count'],'incomplete annotation extraction'
  except (OSError,ValueError,tarfile.TarError,EOFError) as e:pending.append({'route':route,'reason':repr(e)});continue
  samples=[];excluded=[];folder=O/route;folder.mkdir(exist_ok=True)
  for i in range(min(ids)+6,max(ids)-49):
   if not all(j in ann for j in range(i-6,i+51)):excluded.append({'frame':i,'reason':'noncontinuous annotation window'});continue
   motion=np.linalg.norm(np.diff(np.stack([positions[j] for j in range(i-6,i+51)]),axis=0),axis=1).max()
   if motion>8:excluded.append({'frame':i,'reason':'ego_motion_over_8m_per_0.1s','max_m':float(motion)});continue
   d=construct(ann,poses,i);d['source_sha']=np.array(source);d['preprocess_version']=np.array('qwen_v4_anno_only');target=folder/f'{i:05d}.npz';tmp=target.with_suffix('.pending.npz');np.savez_compressed(tmp,**d);tmp.replace(target);samples.append(str(target.relative_to(O)))
  record={**row,'route':route,'file':archive.name,'archive_path':str(archive.resolve()),'split':'train','family':row.get('family',route.split('_Town')[0]),'annotations':len(ids),'annotation_first':min(ids),'annotation_last':max(ids),'annotation_gaps':sorted(set(range(min(ids),max(ids)+1))-set(ids)),'windows':len(samples),'samples':samples,'excluded':excluded,'source_sha256':source};records.append(record);print('BUILT',route,len(samples),flush=True)
  atomic_json(O/'manifest.json',{'routes':records,'pending':pending,'counts':{'train':sum(x['windows'] for x in records)},'coordinate_frame':'current_ego_FLU','frame_time_source':'index*0.1 assumed collection cadence, raw JSON lacks timestamp','parity_report':'parity_report.json','validation_test':'not modified; all new labels training Town12 only','OCC_required':False})
 atomic_json(O/'manifest.json',{'routes':records,'pending':pending,'counts':{'train':sum(x['windows'] for x in records)},'coordinate_frame':'current_ego_FLU','frame_time_source':'index*0.1 assumed collection cadence, raw JSON lacks timestamp','parity_report':'parity_report.json','validation_test':'not modified; all new labels training Town12 only','OCC_required':False});print('DONE',len(records),'pending',len(pending),flush=True)
if __name__=='__main__':main()
