"""Frame-synchronised CARLA chase camera for the fast driver's predictions."""
import math
from queue import Empty, Queue

import carla
import cv2
import numpy as np
from .unified_grid import X,Y,corrected_camera_grid


NAV_BLUE = (255,135,36)
# Nearest first. Draw the nearest slice last where forecasts overlap.
OCC_COLORS = ((53,66,255),(48,139,255),(54,207,255),(234,105,220),(201,133,175))


def unified_layers(rgb,lidar,nav,occ,plane):
    grid,_=corrected_camera_grid(plane)
    pixels=(grid+1)*np.array([320,180],np.float32)-.5
    road=cv2.remap(rgb,pixels,None,cv2.INTER_LINEAR)[:,:,::-1].copy()
    ground=plane[0]*X[None,:]+plane[1]*Y[:,None]+plane[2]
    height=lidar[0]*(4./255.)-.5-ground
    returns=(lidar[1]>0)&(height>.25)
    strength=np.clip(.28+.42*lidar[1]/255.+.30*lidar[2]/255.,0.,1.)
    road[returns]=np.maximum(road[returns],(strength[returns,None]*[235,238,240]).astype(np.uint8))
    opacity=nav[:,:,None].astype(np.float32)*(.78/255.)
    road=(road*(1-opacity)+opacity*NAV_BLUE).astype(np.uint8)
    for mask,color in reversed(list(zip(occ,OCC_COLORS))):
        road[mask!=0]=color
    return road,nav.copy(),occ[0].copy(),tuple(plane)


class ChaseView:
    size = (1920, 1080)
    fov = 78.
    plan_color = (115, 243, 74)
    lane_color = (255, 210, 35)
    occ_color = (65, 83, 255)

    def __init__(self, world, ego, folder, record, incident=False):
        self.folder = folder
        self.writer = None
        self.frame_index = None
        self.queue = Queue()
        self.images = {}
        self.predictions = {}
        self.shown = 0
        self.key_snapshot = False
        bp = world.get_blueprint_library().find("sensor.camera.rgb")
        for k,v in (("image_size_x",self.size[0]),("image_size_y",self.size[1]),("fov",self.fov)):
            bp.set_attribute(k,str(v))
        mount = carla.Transform(carla.Location(x=-9.6,z=5.7),carla.Rotation(pitch=-19))
        self.camera = world.spawn_actor(bp,mount,attach_to=ego)
        try:
            self.camera.listen(self.queue.put)
            if record:
                name,codec=("chase.ts","mp2v") if incident else ("chase.mp4","mp4v")
                self.writer = cv2.VideoWriter(str(folder/name),cv2.VideoWriter_fourcc(*codec),20,self.size)
                if incident:
                    self.frame_index=(folder/"video-frames.txt").open("w",buffering=1)
                if not self.writer.isOpened():raise RuntimeError("Cannot open chase video")
        except Exception:
            self.close()
            raise

    def update(self, frame, pose, result, kmh, layers=None):
        self.predictions[frame] = (pose,result,kmh,layers)
        while True:
            try:image = self.queue.get_nowait()
            except Empty:break
            self.images[image.frame] = image
        for store in (self.images,self.predictions):
            for old in list(store):
                if old < frame-12:del store[old]
        common = sorted(self.images.keys() & self.predictions.keys())
        view = None
        for f in common:
            im = self.images.pop(f)
            pose,result,speed,layers = self.predictions.pop(f)
            view = self.overlay(im,pose,result,speed,layers)
            self.shown += 1
            if self.writer is not None:
                self.writer.write(view)
                if self.frame_index:self.frame_index.write(f"{f}\n")
                if self.shown==1 or (result["intent"] and not self.key_snapshot):
                    if not cv2.imwrite(str(self.folder/"chase.jpg"),view):raise OSError("Cannot save preview")
                    self.key_snapshot = bool(result["intent"])
        return view

    def overlay(self, image, pose, result, speed, layers=None):
        width,height = self.size
        view = np.frombuffer(image.raw_data,np.uint8).reshape(height,width,4)[:,:,:3].copy()
        camera_from_ego = np.asarray(image.transform.get_inverse_matrix()) @ pose
        focal = width/(2*math.tan(math.radians(self.fov)/2))

        def project(xy):
            xy = np.asarray(xy)
            z=.08 if layers is None else .08+xy[:,0]*layers[3][0]+xy[:,1]*layers[3][1]+layers[3][2]
            points = np.column_stack((xy[:,0],-xy[:,1],np.broadcast_to(z,len(xy)),np.ones(len(xy))))
            x,y,z = (points @ camera_from_ego.T)[:,:3].T
            if np.any(x<=.1):return None
            return np.rint(np.column_stack((width/2+focal*y/x,height/2-focal*z/x))).astype(np.int32)

        if layers is not None:
            _,nav,near_occ,_=layers
            # Draw the ridge of the actual quantized corridor on the road. The
            # full input (including behind the car) stays visible in the inset.
            source=np.float32([[20,52],[45,52],[45,76],[20,76]])
            corners=project(np.column_stack((X[source[:,0].astype(int)],Y[source[:,1].astype(int)])))
            if corners is not None:
                transform=cv2.getPerspectiveTransform(source,corners.astype(np.float32))
                for mask,color in ((nav>=218,NAV_BLUE),(near_occ!=0,self.occ_color)):
                    mask[:,:19]=False
                    contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                    for c in contours:
                        # Include single occupied cells at their physical pixel size.
                        if len(c)<3 or cv2.contourArea(c)==0:
                            a,b,w,h=cv2.boundingRect(c)
                            c=np.float32([[a-.5,b-.5],[a+w-.5,b-.5],[a+w-.5,b+h-.5],[a-.5,b+h-.5]]).reshape(-1,1,2)
                        p=cv2.perspectiveTransform(c.astype(np.float32),transform).round().astype(np.int32)
                        cv2.polylines(view,[p],True,color,3,cv2.LINE_AA)
        for row,col in (() if layers is not None else result["occupied_cells"]):
            dy=result.get("grid_dy",.3)
            x,y = col-16,(row-64)*dy
            p = project([[x,y],[x+1,y],[x+1,y+dy],[x,y+dy]])
            if p is not None:cv2.polylines(view,[p],True,self.occ_color,1,cv2.LINE_AA)

        lane = result.get("lane_pose")
        if lane is not None:
            lat,yaw = lane
            tangent = np.array([math.cos(yaw),-math.sin(yaw)])
            normal = np.array([-tangent[1],tangent[0]])
            for s in np.arange(3.,16.,2.):
                pts = project(-lat*normal+np.array([s,s+1.2])[:,None]*tangent)
                if pts is not None:cv2.polylines(view,[pts],False,self.lane_color,3,cv2.LINE_AA)

        plan = np.asarray(result["xy"])
        plan = plan[plan[:,0]>2.7]
        if len(plan)>1:
            p = project(plan)
            if p is not None:cv2.polylines(view,[p],False,self.plan_color,3,cv2.LINE_AA)

        if layers is not None:
            tile=cv2.resize(layers[0].transpose(1,0,2)[::-1,::-1],(216,540),interpolation=cv2.INTER_NEAREST)
            def inset_xy(x,y):
                return np.rint(np.column_stack(((25.6-y)*216/51.2-.5,(112-x)*540/128-.5))).astype(np.int32)
            trajectory=np.asarray(result["xy"])
            cv2.polylines(tile,[inset_xy(*trajectory.T)],False,self.plan_color,2,cv2.LINE_AA)
            car=inset_xy(np.array([2.5,-2.2,-2.2]),np.array([0.,1.1,-1.1]))
            cv2.fillConvexPoly(tile,car,(255,255,255),cv2.LINE_AA)
            x,y=width-232,48
            cv2.rectangle(view,(x-6,12),(x+221,y+602),(35,33,30),-1)
            cv2.putText(view,"BEV   |   FORWARD ^",(x,37),cv2.FONT_HERSHEY_SIMPLEX,.53,(245,245,245),1,cv2.LINE_AA)
            view[y:y+540,x:x+216]=tile
            cv2.putText(view,"Forecast occupancy (s)",(x,y+560),cv2.FONT_HERSHEY_SIMPLEX,.50,(245,245,245),1,cv2.LINE_AA)
            for i,(t,color) in enumerate(zip((".5","1","2","3","4"),OCC_COLORS)):
                dx=x+i*43
                cv2.rectangle(view,(dx,y+572),(dx+9,y+582),color,-1)
                cv2.putText(view,t,(dx+12,y+583),cv2.FONT_HERSHEY_SIMPLEX,.48,(245,245,245),1,cv2.LINE_AA)

        cv2.rectangle(view,(15,14),(351,95),(36,33,29),-1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        intent = ("KEEP","LEFT","RIGHT")[result["intent"]]
        if "fused_keep" in result:
            intent+=" / FUSED" if result["fused_keep"] else " / PLAN"
        if "blue_keep" in result:
            intent+=" / BLUE" if result["blue_keep"] else " / PLAN"
        cv2.putText(view,f"{speed:.0f} km/h    {intent}",(27,44),font,.78,(250,250,250),2,cv2.LINE_AA)
        if lane is not None:
            cv2.putText(view,f"Offset {lat:+.2f} m    Yaw {math.degrees(yaw):+.1f} deg",(27,77),font,.57,
                        self.lane_color,1,cv2.LINE_AA)
        cv2.rectangle(view,(0,height-36),(width,height),(36,33,29),-1)
        for x,label,color in ((18,f"Planned path ({len(result['xy'])/10:g} s)",self.plan_color),
                              (236,"Ch 6 navigation" if layers is not None else "Estimated center tangent",NAV_BLUE if layers is not None else self.lane_color),
                              (640,"Predicted occupancy (0.5 s)" if layers is not None else "Predicted occupancy",self.occ_color)):
            if x==236 and lane is None and layers is None:continue
            cv2.line(view,(x,height-18),(x+25,height-18),color,3,cv2.LINE_AA)
            cv2.putText(view,label,(x+36,height-12),font,.51,(250,250,250),1,cv2.LINE_AA)
        return view

    def close(self):
        self.camera.stop()
        self.camera.destroy()
        if self.writer is not None:self.writer.release()
        if self.frame_index:self.frame_index.close()
