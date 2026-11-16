import sys,json,time,contextlib,argparse
a=argparse.ArgumentParser();a.add_argument('--checkpoint',required=True);args=a.parse_args()
with contextlib.redirect_stdout(sys.stderr):
 import torch,numpy as np
 from train_mixed_regression_flow512 import CompareModel,configure_runtime
 from carla_miniloop_adapter import actor_inputs
 from carla_nav_contract import near_goal
 configure_runtime();ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False);settings=ck.get('task_state',{}).get('settings',{});from scaled_velocity_flow import ScaledVelocityFlow;sc=settings.get('velocity_scales_mps',[10,10]);assert sc[0]==10;m=ScaledVelocityFlow(sc[1]);m.load_state_dict({k[6:]:v for k,v in ck['state_dict'].items() if k.startswith('model.')});m.cuda().eval();state_mode=ck.get('task_state',{}).get('settings',{}).get('state_mode','speed_only');del ck
for line in sys.stdin:
 try:
  req=json.loads(line)
  if req.get('op')!='infer':raise ValueError('infer only')
  frames=req['frames'][-7:];rows=req['rows'][-7:];assert len(frames)==7;ids=[0,2,4,6];raw=[]
  for j in ids:
   with np.load(frames[j]) as z:raw.append({k:z[k].copy() for k in ['lidar_xyz','lidar_tag','lidar_world','ego_world']})
  start=time.perf_counter();b=actor_inputs(raw,[rows[j] for j in ids]);
  if state_mode=='speed_near':b['state'][4:6]=near_goal(rows[6])
  b['geometry_valid']=b['history']!=255;b={k:torch.from_numpy(v)[None].cuda() for k,v in b.items()};torch.manual_seed(2301)
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):p=m.sample(b,k=1,steps=16)[0,0].float()
  torch.cuda.synchronize();assert torch.isfinite(p).all();print(json.dumps({'trajectory':p.cpu().tolist(),'latency_ms':(time.perf_counter()-start)*1000,'source':'fixednoise2301-K1-Euler16-checkpointscales-nav-oracle-semantic-hits'}),flush=True)
 except Exception as e:print(json.dumps({'error':repr(e)}),flush=True)


