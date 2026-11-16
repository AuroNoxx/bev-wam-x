import json,math,time,queue,threading,subprocess
from pathlib import Path
from collections import deque
import numpy as np
from PIL import Image
import carla
from leaderboard.autoagents.autonomous_agent import Track
from b2d_qwen_v4_agent import QwenV4Agent

def get_entry_point():return 'OfficialQwenDriveAgent'

class OfficialQwenDriveAgent(QwenV4Agent):
 def setup(self,path_to_conf_file):
  self.track=Track.SENSORS;self.root=Path(__file__).resolve().parents[1]
  self.out=self.root/'artifacts/qwen-drive-b2d/agent'/str(time.time_ns());self.out.mkdir(parents=True)
  self.history=deque(maxlen=80);self.index=0;self.tick=0;self.integral=0.;self.control=carla.VehicleControl(brake=1)
  self.err=(self.out/'worker.stderr.log').open('w',encoding='utf8')
  self.worker=subprocess.Popen([str(self.root/'.venv-train/Scripts/python.exe'),'-u',str(self.root/'scripts/qwen_drive_b2d_worker.py')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.err,text=True,encoding='utf8')
  self.responses=queue.Queue()
  def reader():
   for line in self.worker.stdout:self.responses.put(line)
   self.responses.put(None)
  threading.Thread(target=reader,daemon=True).start()
  if not self.read_response(240).get('ready'):raise RuntimeError('worker failed')
  self.log=(self.out/'steps.jsonl').open('w',encoding='utf8')
 def sensors(self):
  base=super().sensors()
  for id,yaw in [('left',-60.),('right',60.)]:
   c=base[0].copy();c.update(id=id,yaw=yaw);base.append(c)
  return base
 def run_step(self,input_data,timestamp):
  self.tick+=1
  imu=np.asarray(input_data['imu'][1]);gps=input_data['gps'][1];speed=float(input_data['speed'][1]['speed'])
  pos=self.project(float(gps[0]),float(gps[1]));yaw=float(imu[-1])-math.pi/2
  images={}
  if self.tick%2==0:
   for id in ('rgb','left','right'):
    p=self.out/f'{id}-{self.tick%80:02d}.jpg';Image.fromarray(input_data[id][1][:,:,:3][:,:,::-1]).save(p,quality=95);images[id]=str(p)
  self.history.append(dict(t=float(timestamp),pos=pos,yaw=yaw,speed=speed,acc=[float(imu[0]),float(-imu[1])],images=images))
  if len(self.history)<32 or self.tick%4:return self.control
  h=list(self.history)[-31::2];assert len(h)==16
  if any(abs(h[i+1]['t']-h[i]['t']-.1)>.01 for i in range(15)):raise RuntimeError('history timing')
  rot=np.array([[math.cos(yaw),math.sin(yaw)],[math.sin(yaw),-math.cos(yaw)]])
  poses=[[*list(rot@(v['pos']-pos)),float((yaw-v['yaw']+math.pi)%(2*math.pi)-math.pi)] for v in h]
  self.index=min(range(self.index,min(self.index+50,len(self.route))),key=lambda i:np.linalg.norm(self.route[i][0]-pos))
  k=self.index;length=0.
  while k+1<len(self.route) and length<30:
   length+=np.linalg.norm(self.route[k+1][0]-self.route[k][0]);k+=1
  command=self.route[k][1];nav=1 if command in (1,5) else 2 if command in (2,6) else 0
  driving=[float(nav==1),float(nav==0),float(nav==2),0.]
  views={tag:[h[i]['images'][id] for i in (0,5,10,15)] for tag,id in [('<FRONT VIEW>','rgb'),('<FRONT LEFT VIEW>','left'),('<FRONT RIGHT VIEW>','right')]}
  # Official interpolation/data.md use per-timestamp body-frame dynamics.
  a=dict(views=views,history=poses,history_velocity=[[v['speed'],0.] for v in h],history_acceleration=[v['acc'] for v in h],ego_velocity=[speed,0.],ego_acceleration=h[-1]['acc'],driving_command=driving,nav_command=nav)
  self.worker.stdin.write(json.dumps(a)+'\n');self.worker.stdin.flush();out=self.read_response(180)
  if 'error' in out:raise RuntimeError(out['error'])
  xy=np.asarray(out['trajectory'])[:,:2];target=xy[9]
  target_speed=float(xy[1,0]/.2);error=target_speed-speed;candidate=self.integral+.2*error
  if abs(.2*error+.1*candidate)<=1 or (.2*error+.1*candidate)*error<0:self.integral=candidate
  longitudinal=.2*error+.1*self.integral
  steer=-math.atan2(2*2.8*target[1],max(float(target@target),.01))/math.radians(70)
  self.control=carla.VehicleControl(steer=float(np.clip(steer,-1,1)),throttle=float(np.clip(longitudinal,0,1)),brake=float(np.clip(-longitudinal,0,1)))
  self.log.write(json.dumps(dict(timestamp=timestamp,speed=speed,nav=nav,history=poses,output=out,control=dict(steer=self.control.steer,throttle=self.control.throttle,brake=self.control.brake)))+'\n');self.log.flush()
  return self.control
