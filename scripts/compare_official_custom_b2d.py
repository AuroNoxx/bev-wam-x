"""Inference-only native-input paired B2D smoke comparison; no training."""
from pathlib import Path
import sys,json,time,gzip,tarfile,io,hashlib
R=Path(__file__).resolve().parents[1];O=R/'artifacts/qwen-drive-paired-b2d';O.mkdir(exist_ok=True)
stage=sys.argv[1]
if stage=='prepare':
 import numpy as np,torch
 from PIL import Image
 from torch.nn.attention import sdpa_kernel,SDPBackend
 from qwen_driving_v3_head import Head
 torch.set_num_threads(6)
 names=['HighwayCutIn_Town12_Route13451_Weather1_02-06-10-18-37','HighwayExit_Town12_Route23687_Weather11_02-06-10-18-47','HighwayCutIn_Town13_Route23644_Weather13_02-07-19-13-33','HighwayExit_Town13_Route23821_Weather16_02-06-22-04-16']
 ck=R/'artifacts/qwen-driving-v4/weights/best.ckpt';model=Head('joint_fm').cuda().eval();model.load_state_dict(torch.load(ck,map_location='cpu',weights_only=False)['state_dict']);model.requires_grad_(False);out=[]
 for k,name in enumerate(names):
  rows=torch.load(R/'artifacts/qwen-driving-v3/cache'/(name+'.pt'),weights_only=False)
  ann={int(p.name.split('.')[0]):json.load(gzip.open(p,'rt')) for p in (R/'data/public/bench2drive/clips'/name/'anno').glob('*.json.gz')}
  eligible=[r for r in rows if r['meta']['frame']>=16 and all(i in ann for i in range(r['meta']['frame']-16,r['meta']['frame']+51)) and ann[r['meta']['frame']]['command_near']==3 and ann[r['meta']['frame']]['command_far']==3]
  assert eligible,name
  r=eligible[len(eligible)//2];i=r['meta']['frame'];folder=O/str(k);folder.mkdir(exist_ok=True)
  w2e={j:np.array(next(b for b in a['bounding_boxes'] if b['class']=='ego_vehicle')['world2ego']) for j,a in ann.items()};e2w={j:np.linalg.inv(m) for j,m in w2e.items()}
  def pose(j):
   m=w2e[i]@e2w[j];return [m[0,3],-m[1,3],-np.arctan2(m[1,0],m[0,0])]
  hist=np.array([pose(j) for j in range(i-15,i+1)]);hist[-1]=0
  gt=np.array([pose(j) for j in range(i+1,i+51)])
  vel=np.array([(w2e[j][:3,:3]@((e2w[j][:3,3]-e2w[j-1][:3,3])/.1))[:2]*[1,-1] for j in range(i-15,i+1)])
  acc=np.array([[ann[j]['acceleration'][0],-ann[j]['acceleration'][1]] for j in range(i-15,i+1)])
  cached=r['trajectory'][:,:2].numpy()*[50,10];parity=float(np.max(abs(cached-gt[1::2,:2])));assert parity<1e-4,parity
  ids=[i-15,i-10,i-5,i];imgs={};dims={}
  with tarfile.open(R/'data/public/bench2drive/archives'/(name+'.tar.gz'),'r|gz') as tar:
   for entry in tar:
    pp=Path(entry.name)
    if entry.isfile() and pp.parent.name in ['rgb_front','rgb_front_left','rgb_front_right'] and int(pp.stem) in ids:
     data=tar.extractfile(entry).read();dest=folder/(pp.parent.name+'_'+pp.name);dest.write_bytes(data);imgs[pp.parent.name+':'+pp.stem]=str(dest);dims[pp.parent.name]=Image.open(io.BytesIO(data)).size
  assert len(imgs)==12,len(imgs)
  np.savez(folder/'inputs.npz',history=hist,velocity=vel,acceleration=acc,ground_truth=gt)
  f=r['features'][None].cuda();st=r['state'][None].cuda();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.perf_counter()
  with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
   g=torch.Generator(device='cuda').manual_seed(9188);x=torch.randn((1,25,2),device='cuda',generator=g);w=torch.randn((1,16,32),device='cuda',generator=g)
   for j in range(16):
    a,v=model(f,st,x,torch.full((1,1),j/16,device='cuda'),w);x=x+a/16;w=w+v/16
   ego=torch.stack([st[:,0]*30,torch.zeros(1,device='cuda')],-1)[:,None,:];pred=((x*10+ego).cumsum(1)*.2)[0].cpu().numpy()
  torch.cuda.synchronize();elapsed=time.perf_counter()-t;err=np.linalg.norm(pred-cached,axis=-1)
  item=dict(case=k,route=name,frame=i,split=r['meta']['split'],native_image_size=dims,official_image_ids=ids,custom_image_ids=[i-6,i-4,i-2,i],command_near=3,command_far=3,official_nav=0,command_source='current annotation only; no future-derived navigation',label_parity_max_m=parity,speed_mps=ann[i]['speed'],backward_difference_speed_mps=float(np.linalg.norm(vel[-1])),speed_history_max_discrepancy_mps=float(max(abs(np.linalg.norm(v)-ann[j]['speed']) for v,j in zip(vel,range(i-15,i+1)))),custom=dict(trajectory=pred.tolist(),ADE_m=float(err.mean()),FDE_m=float(err[-1]),head_only_seconds=elapsed,seed=9188),ground_truth_5hz=cached.tolist())
  out.append(item);print(json.dumps({key:value for key,value in item.items() if key not in ['ground_truth_5hz','custom']}),flush=True)
 (O/'cases.json').write_text(json.dumps(out,indent=2))
 (O/'protocol.json').write_text(json.dumps(dict(training=False,custom_checkpoint=str(ck),custom_sha256=hashlib.sha256(ck.read_bytes()).hexdigest(),selection='four preselected train/validation routes; median eligible cached window with current near=far=STRAIGHT(3), complete history+future; no prediction-dependent selection',metrics='both 25 XY positions at 0.2..5.0 seconds; official select indices 1,3,...49',native_inputs_differ=True,velocity='causal backward difference of measured world poses projected into each timestamp body FLU',acceleration='measured annotation IMU x,-y at each timestamp, matching released Waymo per-body dynamic convention',limitations=['native input and navigation interfaces differ','Town13 validation used for best-checkpoint selection, not untouched test','image and sensor domain shift','frame rate assumed 10 Hz per B2D contract, raw timestamp absent','backward-difference velocity vs direct sensors','straight-only small selected probe; no safety/generalization claim']),indent=2))
elif stage=='official':
 sys.path[:0]=[str(R/'runtime/qwen-drive-deps'),str(R/'runtime/qwen-drive-official/src')]
 import numpy as np,torch
 from qwen_drive import QwenDriveForPlanning,InferenceMode
 from qwen_drive.scene import DrivingScene,CameraFrame,CAMERA_VIEWS
 from torch.nn.attention import sdpa_kernel,SDPBackend
 torch.set_num_threads(6);M=R/'models/Qwen-Drive-1.0-4B';q=QwenDriveForPlanning.from_pretrained(M,planner=M/'planner-sft',dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).cuda().eval();q.requires_grad_(False)
 cases=json.loads((O/'cases.json').read_text());results=[]
 for c in cases:
  k=c['case'];folder=O/str(k);z=np.load(folder/'inputs.npz');views={key:[CameraFrame(folder/(camera+'_'+f'{i:05d}.jpg')) for i in c['official_image_ids']] for key,camera in zip(CAMERA_VIEWS,['rgb_front','rgb_front_left','rgb_front_right'])}
  scene=DrivingScene(views=views,history=z['history'],history_velocity=z['velocity'],history_acceleration=z['acceleration'],ego_velocity=z['velocity'][-1],ego_acceleration=z['acceleration'][-1],driving_command=[0,1,0,0],nav_command=0,token=c['route']+':'+str(c['frame']))
  inputs=q.processor(scene,with_reasoning=False,device='cpu');torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.perf_counter();torch.manual_seed(9188)
  with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):result=q.plan_from_inputs(inputs,mode=InferenceMode.DIRECT_PLANNING,num_samples=1)
  torch.cuda.synchronize();elapsed=time.perf_counter()-t;pred=result.trajectory;assert pred.shape==(50,3) and np.isfinite(pred).all();err=np.linalg.norm(pred[1::2,:2]-np.array(c['ground_truth_5hz']),axis=-1)
  c['official']=dict(trajectory=pred.tolist(),ADE_m=float(err.mean()),FDE_m=float(err[-1]),seconds=elapsed,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,input_tokens=int(inputs['input_ids'].shape[1]),image_grid=inputs['image_grid_thw'].tolist(),seed=9188,num_samples=1)
  results.append(c);(O/'outputs.json').write_text(json.dumps(results,indent=2));print(json.dumps(dict(case=k,split=c['split'],custom_ADE=c['custom']['ADE_m'],custom_FDE=c['custom']['FDE_m'],official_ADE=c['official']['ADE_m'],official_FDE=c['official']['FDE_m'],seconds=elapsed)),flush=True)
 summary={s:{model:{metric:float(np.mean([c[model][metric] for c in results if c['split']==s])) for metric in ['ADE_m','FDE_m']} for model in ['custom','official']} for s in ['train','validation']};summary['complete']=True;summary['training']=False;(O/'completed.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
