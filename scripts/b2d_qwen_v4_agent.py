"""Frozen Qwen-v4 sensor-only adapter. No simulator actor queries in policy."""
import json, math, os, subprocess, time, queue, threading
from pathlib import Path
from collections import deque
import numpy as np
from PIL import Image
import carla
from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track

def get_entry_point(): return 'QwenV4Agent'

class QwenV4Agent(AutonomousAgent):
    def setup(self, path_to_conf_file):
        self.track=Track.SENSORS
        self.root=Path(__file__).resolve().parents[1]
        self.out=self.root/'artifacts/b2d-0915/agent'/str(time.time_ns())
        self.out.mkdir(parents=True,exist_ok=True)
        self.history=deque(maxlen=32); self.index=0; self.tick=0; self.integral=0.
        self.control=carla.VehicleControl(brake=1)
        self.err=(self.out/'worker.stderr.log').open('w')
        self.worker=subprocess.Popen([str(self.root/'.venv-train/Scripts/python.exe'),'-u',str(self.root/'scripts/qwen_carla_worker_v4.py')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.err,text=True)
        self.responses=queue.Queue()
        def reader():
            for line in self.worker.stdout:self.responses.put(line)
            self.responses.put(None)
        threading.Thread(target=reader,daemon=True).start()
        if not self.read_response(180).get('ready'): raise RuntimeError('Worker not ready')
        self.log=(self.out/'steps.jsonl').open('w')
    def read_response(self,timeout=60):
        try: line=self.responses.get(timeout=timeout)
        except queue.Empty: raise TimeoutError('Qwen worker response timed out')
        if line is None:raise RuntimeError('Qwen worker exited; inspect worker.stderr.log')
        return json.loads(line)
    def sensors(self):
        return [dict(type='sensor.camera.rgb',x=.8,y=0.,z=1.6,roll=0.,pitch=0.,yaw=0.,width=1600,height=900,fov=70,id='rgb'),dict(type='sensor.other.imu',x=0.,y=0.,z=0.,roll=0.,pitch=0.,yaw=0.,id='imu'),dict(type='sensor.other.gnss',x=0.,y=0.,z=0.,id='gps'),dict(type='sensor.speedometer',reading_frequency=20,id='speed')]
    def set_global_plan(self,gps,world):
        super().set_global_plan(gps,world)
        # GPS reference route is allowed navigation input; no world pose/map query.
        self.lat0=float(gps[0][0]['lat']);self.lon0=float(gps[0][0]['lon'])
        self.route=[(self.project(p['lat'],p['lon']),int(c.value)) for p,c in gps]
    def project(self,lat,lon):
        # CARLA Mercator world x east, y south, with local scale cos(reference latitude).
        scale=math.cos(math.radians(self.lat0));R=6378137.
        return np.array([scale*R*math.radians(lon-self.lon0),-scale*R*(math.log(math.tan(math.pi/4+math.radians(lat)/2))-math.log(math.tan(math.pi/4+math.radians(self.lat0)/2)))])
    def run_step(self,input_data,timestamp):
        self.tick+=1
        frame,rgb=input_data['rgb'];p=self.out/f'rgb-{self.tick%32:02d}.png'
        # Only sampled 5Hz frames are read by the model; retain exact tick/time selection.
        if self.tick%4==0: Image.fromarray(rgb[:,:,:3][:,:,::-1]).save(p)
        self.history.append((float(timestamp),str(p)))
        if len(self.history)<13 or self.tick%4:return self.control
        imu=np.asarray(input_data['imu'][1]);gps=input_data['gps'][1];speed=float(input_data['speed'][1]['speed'])
        if imu.shape!=(7,) or not np.isfinite(imu).all() or not np.isfinite(gps).all() or not math.isfinite(speed):raise RuntimeError('Invalid sensor values')
        pos=self.project(float(gps[0]),float(gps[1]));yaw=float(imu[-1])-math.pi/2
        rot=np.array([[math.cos(yaw),math.sin(yaw)],[-math.sin(yaw),math.cos(yaw)]])
        self.index=min(range(self.index,min(self.index+50,len(self.route))),key=lambda i:np.linalg.norm(self.route[i][0]-pos))
        goals=[];commands=[]
        for distance in (5.,30.):
            k=self.index;length=0.
            while k+1<len(self.route) and length<distance:
                length+=np.linalg.norm(self.route[k+1][0]-self.route[k][0]);k+=1
            local=rot@(self.route[k][0]-pos);goals.extend([float(local[0]/50),float(-local[1]/50)]);commands.append(self.route[k][1])
        state=[speed/30,float(imu[0]/10),float(-imu[1]/10),float(-imu[5])]+goals+[float(c==j) for c in commands for j in range(1,7)]
        hist=[self.history[-13],self.history[-9],self.history[-5],self.history[-1]]
        if any(abs(hist[i+1][0]-hist[i][0]-.2)>.025 for i in range(3)):raise RuntimeError('History time contract violated')
        self.worker.stdin.write(json.dumps(dict(images=[x[1] for x in hist],state=state))+'\n');self.worker.stdin.flush()
        out=self.read_response()
        if 'error' in out:raise RuntimeError(out['error'])
        xy=np.asarray(out['xy']);vel=np.asarray(out['velocity'])
        if xy.shape!=(25,2) or vel.shape!=(25,2) or not np.isfinite(xy).all() or not np.isfinite(vel).all():raise RuntimeError('Invalid trajectory')
        target=xy[4];error=float(vel[0,0])-speed;candidate=self.integral+.2*error
        if abs(.2*error+.1*candidate)<=1 or (.2*error+.1*candidate)*error<0:self.integral=candidate
        longitudinal=.2*error+.1*self.integral
        # Fixed vehicle geometry, not actor/physics access. Must be disclosed and calibrated separately.
        steer=-math.atan2(2*2.8*target[1],max(float(target@target),.01))/math.radians(70)
        self.control=carla.VehicleControl(steer=float(np.clip(steer,-1,1)),throttle=float(np.clip(longitudinal,0,1)),brake=float(np.clip(-longitudinal,0,1)))
        self.log.write(json.dumps(dict(timestamp=timestamp,state20=state,output=out,control=dict(steer=self.control.steer,throttle=self.control.throttle,brake=self.control.brake)))+'\n');self.log.flush()
        return self.control
    def destroy(self):
        if hasattr(self,'worker'):
            self.worker.terminate();self.worker.wait(timeout=15)
        for name in ('err','log'):
            if hasattr(self,name):getattr(self,name).close()
