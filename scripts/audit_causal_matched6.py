import json,hashlib,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]/'data/carla_causal_matched6_20261116_v1'
def spans(mask):
 ids=np.where(mask)[0]
 return [[int(x[0]),int(x[-1])] for x in np.split(ids,np.where(np.diff(ids)>1)[0]+1) if len(x)]
def main():
 mf=json.loads((ROOT/'manifest.json').read_text());out=[]
 for ep in mf['episodes']:
  folder=ROOT/f"episode{ep['episode']}";rs=json.loads((folder/'frames.json').read_text());hashes={};hit=[];maxnav=0
  for r in rs:
   assert r['frame']==r['rgb_frame']==r['lidar_frame']==r['teacher_target_post_audit']['snapshot_frame']==r['teacher_observation_frame']+1
   assert r['teacher_target_pre_audit']['snapshot_frame']==r['teacher_observation_frame']
   assert abs(r['timestamp']-r['teacher_observation_timestamp']-.1)<1e-5
   assert r['command_current']['movement_permitted'] and r['command_current']['cruise_mps']==ep['target_speed_mps']
   path=folder/f"{r['index']:05}.npz";hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
   with np.load(path) as z:
    assert z['rgb_bgra'].shape==(360,640,4);assert np.max(abs(z['ego_world']-r['ego_world']))==0
    assert len(z['lidar_xyz'])==len(z['lidar_tag'])==len(z['lidar_object_idx']);hit.append(int(sum(z['lidar_object_idx']==r['teacher_target_post_audit']['actor_id'])))
   v=np.linalg.inv(np.array(r['ego_world']))@np.r_[r['near_goal_world'],1];maxnav=max(maxnav,float(np.max(abs(v[:3]-r['near_goal_local_carla']))))
  (folder/'sha256.json').write_text(json.dumps(hashes))
  es=np.array([np.linalg.norm(r['velocity_world'][:2]) for r in rs]);ls=np.array([np.linalg.norm(r['teacher_target_post_audit']['velocity_world'][:2]) for r in rs]);leadstops=spans(ls<.05);egostops=spans(es<.05);restarts=[i for i in range(1,len(rs)) if ls[i]>.2 and ls[i-1]<=.2];details=[]
  for i in restarts:
   eligible=[j for j in range(i,min(i+50,len(rs))) if es[j]>.2];details.append({'lead_restart_index':i,'lead_restart_seconds':i*.1,'ego_above_point2_index':eligible[0] if eligible else None,'ego_speed_3s_later':float(es[min(i+30,len(rs)-1)]),'ego_control_applied_at_next_tick':rs[min(i+1,len(rs)-1)]['control_applied']})
  ttc=[r['teacher_inputs']['ttc_s'] for r in rs if r['teacher_inputs']['ttc_s'] is not None]
  out.append({'episode':ep['episode'],'split':ep['split'],'road':ep['road'],'seed':ep['seed'],'status':ep['status'],'guard':ep['guard'],'frames':len(rs),'cruise_mps':ep['target_speed_mps'],'initial_gap_m':ep['initial_relative']['gap_m'],'candidate_windows_history6_future50':max(0,len(rs)-56),'compatibility_windows_start36':max(0,len(rs)-86),'min_gap_m':min(r['teacher_inputs']['gap_m'] for r in rs),'min_ttc_s':min(ttc) if ttc else None,'min_bbox_margin_m':min(r['bbox_lane_margin_m'] for r in rs),'collision_callbacks':ep['collision_callbacks'],'max_speed_mps':float(max(es)),'final_speed_mps':float(es[-1]),'lead_stop_index_ranges':leadstops,'ego_stop_index_ranges':egostops,'actual_lead_restart_response':details,'lead_visible_raw_frames':sum(v>0 for v in hit),'lead_hits_min_median_max':np.quantile(hit,[0,.5,1]).tolist(),'nav_local_transform_max_error_m':maxnav})
 report={'episodes':out,'claims':'All event ranges use measured speeds<0.05m/s; restarts use actual lead crossing0.2m/s. No success/quality selection or future data in actor. Candidate windows not final adapter acceptance. Collision callback coverage is registered but not calibrated.','command_alignment':'command_current uses containing row frame/timestamp; teacher observes f then applied control leads to raw f+1','original_guards_unchanged':True,'training_started':False}
 (ROOT/'integrity_event_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
