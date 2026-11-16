"""Approved16 Town12 B2D expansion, pinned files with SHA256; no train split edits."""
from pathlib import Path,PurePosixPath
from concurrent.futures import ThreadPoolExecutor,as_completed
import json,urllib.request,re,hashlib,subprocess,time,threading,tarfile,gzip
R=Path(__file__).resolve().parents[1];O=R/'artifacts/b2d-expansion-v4';O.mkdir(exist_ok=True);DEST=R/'data/public/bench2drive/archives';CLIPS=R/'data/public/bench2drive/clips_v4';REV='d84ae18cf20a88b9220df1bc15c073d8824b7ac1';BASE='https://huggingface.co/datasets/rethinklab/Bench2Drive-V0.0.4/resolve/'+REV+'/'
u='https://huggingface.co/api/datasets/rethinklab/Bench2Drive-V0.0.4/tree/'+REV+'?recursive=false&limit=1000';allrows=[]
while u:
 r=urllib.request.urlopen(u,timeout=30);allrows+=json.load(r);m=re.search(r'<([^>]+)>; rel="next"',r.headers.get('Link',''));u=m.group(1) if m else None
families=['ParkingExit','VanillaSignalizedTurnEncounterRedLight','VanillaNonSignalizedTurnEncounterStopsign','SignalizedJunctionLeftTurn','SignalizedJunctionRightTurn','BlockedIntersection','HardBreakRoute','SequentialLaneChange'];local={p.name for p in DEST.glob('*.tar.gz')};tasks=[]
for fam in families:
 rr=sorted([r for r in allrows if r['path'].startswith(fam+'_Town12_') and r['path'] not in local],key=lambda r:abs(r['size']-350000000));used=set()
 for r in rr:
  rid=r['path'].split('_')[2]
  if rid in used:continue
  used.add(rid);tasks.append(dict(file=r['path'],route=r['path'][:-7],family=fam,town='Town12',source=BASE+r['path'],revision=REV,expected_bytes=r['size'],expected_sha256=r['lfs']['oid'],status='planned'))
  if len(used)==2:break
assert len(tasks)==16 and sum(t['expected_bytes'] for t in tasks)==5115567084
(O/'plan.json').write_text(json.dumps(tasks,indent=2));lock=threading.Lock();started=time.time()
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def update(t,**kw):
 with lock:
  t.update(kw);(O/'manifest.json').write_text(json.dumps(dict(revision=REV,started=started,updated=time.time(),items=tasks),indent=2))
def work(t):
 p=DEST/t['file'];part=p.with_name(p.name+'.partial');update(t,status='downloading')
 try:
  if not p.exists():
   r=subprocess.run(['curl.exe','-L','--fail','--retry','2','--retry-delay','2','--connect-timeout','20','--max-time','600','-sS','-C','-','-o',str(part),t['source']],capture_output=True,text=True)
   if r.returncode:raise RuntimeError('curl:'+r.stderr[-600:])
   if part.stat().st_size!=t['expected_bytes'] or digest(part)!=t['expected_sha256']:raise RuntimeError('download size/SHA256 mismatch')
   if p.exists():raise RuntimeError('destination appeared; no overwrite')
   part.rename(p)
  if p.stat().st_size!=t['expected_bytes'] or digest(p)!=t['expected_sha256']:raise RuntimeError('existing archive failed verification')
  update(t,status='archive_verified',bytes=p.stat().st_size,sha256=t['expected_sha256']);count=0;speeds=[];target=CLIPS/t['route']/'anno';target.mkdir(parents=True,exist_ok=True)
  with tarfile.open(p,'r|gz') as tf:
   for z in tf:
    pp=PurePosixPath(z.name)
    if len(pp.parts)==3 and pp.parts[0]==t['route'] and pp.parts[1]=='anno' and re.fullmatch(r'\d+\.json\.gz',pp.name) and z.isfile():
     raw=tf.extractfile(z).read();a=json.loads(gzip.decompress(raw));dest=target/pp.name
     if dest.exists():
      if dest.read_bytes()!=raw:raise RuntimeError('annotation exists with different content')
     else:dest.write_bytes(raw)
     count+=1
     if isinstance(a.get('speed'),(int,float)):speeds.append(float(a['speed']))
  update(t,status='complete',annotation_count=count,annotation_directory=str(target),speed_summary_mps=dict(min=min(speeds) if speeds else None,max=max(speeds) if speeds else None,mean=sum(speeds)/len(speeds) if speeds else None,under_0_2=sum(abs(x)<.2 for x in speeds),over_15=sum(x>15 for x in speeds)),elapsed_seconds=time.time()-started)
  print('COMPLETE',t['file'],count,flush=True)
 except Exception as e:update(t,status='failed',error=repr(e));print('FAILED',t['file'],repr(e),flush=True)
with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(work,tasks))
print('ALL_DONE',sum(t['status']=='complete' for t in tasks),'/',len(tasks),time.time()-started,flush=True)
