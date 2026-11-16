"""Extract unchanged native B2D RGB for two heldout windows; past/current only."""
from pathlib import Path
import json,tarfile,io,gzip,hashlib
from PIL import Image
R=Path(__file__).resolve().parents[1];O=R/'artifacts/qwen-original-b2d-inputs';O.mkdir(exist_ok=True)
plan=json.loads((R/'artifacts/qwen-driving-v3/selection.json').read_text());chosen=[]
for family in ['HighwayCutIn','HardBreakRoute']:
 route=next(x['route'] for x in plan if x['split']=='validation' and x['family']==family);rows=[x for x in plan if x['route']==route];chosen.append(rows[len(rows)//2])
reports=[]
for sample in chosen:
 route=sample['route'];anchor=sample['frame'];history=sample['history'];archive=R/'data/public/bench2drive/archives'/(route+'.tar.gz');dest=O/route;dest.mkdir(exist_ok=True);inventory={};files=[]
 with tarfile.open(archive,'r|gz') as tar:
  for member in tar:
   parts=Path(member.name).parts
   if not member.isfile() or 'camera' not in parts:continue
   j=parts.index('camera')
   if j+2>=len(parts):continue
   camera=parts[j+1]
   if not camera.startswith('rgb_'):continue
   try:frame=int(Path(member.name).stem)
   except ValueError:continue
   inventory.setdefault(camera,set()).add(frame)
   if not ((camera=='rgb_front' and frame in history) or frame==anchor):continue
   assert frame<=anchor
   data=tar.extractfile(member).read();im=Image.open(io.BytesIO(data));target=dest/camera/Path(member.name).name;target.parent.mkdir(exist_ok=True);target.write_bytes(data)
   files.append({'camera':camera,'frame':frame,'archive_member':member.name,'extracted_path':str(target.resolve()),'native_width':im.width,'native_height':im.height,'format':im.format,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'raw_bytes_unchanged':True})
 annos=[]
 for frame in history:
  path=R/'data/public/bench2drive/clips'/route/'anno'/f'{frame:05d}.json.gz';a=json.load(gzip.open(path,'rt'));timekeys=[k for k in a if 'timestamp' in k.lower() or k.lower() in ['time','time_s','frame','frame_id']]
  annos.append({'frame':frame,'annotation_path':str(path),'annotation_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'root_time_or_frame_keys':{k:a[k] for k in timekeys},'sensor_keys':list(a.get('sensors',{})),'camera_calibration_at_anchor':a.get('sensors',{}) if frame==anchor else None})
 front=inventory['rgb_front'];assert set(history)<=front
 report={'route':route,'family':sample['family'],'split':'validation','anchor':anchor,'history_frames':history,'nominal_history_offsets_s':[(f-anchor)*.1 for f in history],'files':files,'camera_inventory':{k:{'count':len(v),'first':min(v),'last':max(v),'anchor_present':anchor in v,'all_indices_equal_front':v==front,'missing_vs_front':sorted(front-v),'extra_vs_front':sorted(v-front)} for k,v in inventory.items()},'annotations':annos,'time_alignment_limit':'Same numbered files and matching index sets support pairing, not independently verified sensor timestamp synchrony.0.1s source cadence is assumed; no future frames extracted.'};reports.append(report);(dest/'manifest.json').write_text(json.dumps(report,indent=2));print(route,'anchor',anchor,'cameras',len(inventory),'selectedfiles',len(files),[(f['camera'],f['native_width'],f['native_height']) for f in files if f['frame']==anchor],flush=True)
(O/'manifest.json').write_text(json.dumps(reports,indent=2));print('DONE',len(reports),flush=True)
