"""Pure mock sensor contract tests: no CARLA server or model loading."""
import sys,types,math,json,tempfile,io
from pathlib import Path
from collections import deque
import numpy as np
m=types.ModuleType('carla')
class Control:
 def __init__(self,steer=0,throttle=0,brake=0):self.steer=steer;self.throttle=throttle;self.brake=brake
m.VehicleControl=Control;sys.modules['carla']=m
m=types.ModuleType('leaderboard.autoagents.autonomous_agent')
class Base:
 def set_global_plan(self,*a):pass
m.AutonomousAgent=Base;m.Track=types.SimpleNamespace(SENSORS='SENSORS');sys.modules['leaderboard.autoagents.autonomous_agent']=m
from b2d_qwen_v4_agent import QwenV4Agent
for yaw in (0,math.pi/2,math.pi,-math.pi/2):
 with tempfile.TemporaryDirectory() as td:
  a=QwenV4Agent.__new__(QwenV4Agent);a.out=Path(td);a.lat0=0.;a.lon0=0.;a.tick=0;a.index=0;a.integral=0.;a.history=deque(maxlen=32);a.control=Control(brake=1);a.log=io.StringIO()
  forward=np.array([math.cos(yaw),math.sin(yaw)])
  a.route=[(forward*i,4) for i in range(100)]
  a.worker=types.SimpleNamespace(stdin=io.StringIO());a.read_response=lambda:dict(xy=[[i+1,0] for i in range(25)],velocity=[[3,0]]*25)
  imu=np.array([1,2,0,0,0,.3,yaw+math.pi/2])
  data={'rgb':(1,np.zeros((10,10,4),dtype=np.uint8)),'gps':(1,np.zeros(3)),'imu':(1,imu),'speed':(1,{'speed':0.})}
  for i in range(16):c=a.run_step(data,i*.05)
  request=json.loads(a.worker.stdin.getvalue());st=request['state'];assert len(st)==20
  assert np.allclose(st[:4],[0,.1,-.2,-.3]);assert np.allclose(st[4:8],[.1,0,.6,0],atol=1e-8)
  assert st[11]==1 and st[17]==1 and c.throttle>0 and abs(c.steer)<1e-8
  assert len(request['images'])==4
  a.read_response=lambda:dict(xy=[[i+1,1] for i in range(25)],velocity=[[3,0]]*25)
  for i in range(16,20):c=a.run_step(data,i*.05)
  assert c.steer<0,'FLU left must become negative CARLA steering'
  east=a.project(0,.001);north=a.project(.001,0);assert east[0]>0 and north[1]<0
print('PASS: four headings, 20D normalization, command one-hot, history, positive throttle, left steering and GPS axis signs; no model/server used')
