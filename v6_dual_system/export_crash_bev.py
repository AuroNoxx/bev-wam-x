"""Export collision-relative evidence from a capture, entirely offline."""
import argparse
import csv
import json
from pathlib import Path
import subprocess

import carla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
import numpy as np
from scipy.spatial import cKDTree

from .controller import plan_curvature
from .sensors import ROOT,ego_points
from .unified_grid import X,Y


def first_crossing(points,x):
    for start,end in zip(points,points[1:]):
        if start[0]<=x<=end[0] and end[0]>start[0]:
            return float(np.interp(x,[start[0],end[0]],[start[1],end[1]]))
    return None


def ridge(nav,route):
    result=[]
    for i,x in enumerate(X):
        if x<=0:continue
        nominal=first_crossing(route,x)
        if nominal is None:continue
        near=np.flatnonzero((abs(Y-nominal)<2.) & (nav[:,i]>245))
        if len(near):
            top=near[nav[near,i]==nav[near,i].max()]
            result.append([x,np.mean(Y[top])])
    return np.asarray(result)


def aligned_returns(d,meta):
    point=ego_points(d["scan"])*[1.,-1.,1.]
    semantic=d["semantic_xyz"]
    if not len(semantic):return np.empty((0,3)),np.empty(0)
    dist,i=cKDTree(semantic).query(point,distance_upper_bound=.002)
    valid=np.isfinite(dist)
    point=point[valid]
    tags=d["semantic_tags"][i[valid]]
    height=point[:,2]-(point[:,:2]@d["plane"][:2]+d["plane"][2])
    # Only returns at a height that overlaps the actual vehicle.
    visible=(height>=.05)&(height<=meta["box_center"][2]+meta["extent"][2])
    return point[visible],tags[visible]


def clearance(points,centre,yaw,extent):
    if not len(points):return None
    c,s=np.cos(yaw),np.sin(yaw)
    offset=(points[:,:2]-centre)@np.array([[c,-s],[s,c]])
    return float(np.linalg.norm(np.maximum(np.abs(offset)-extent,0.),axis=1).min())


def driving(points,d,road):
    z=points@d["plane"][:2]+d["plane"][2]
    local=np.column_stack((points[:,0],-points[:,1],z,np.ones(len(z))))
    world=local@d["sensor_pose"].T
    return np.asarray([road.get_waypoint(carla.Location(*p[:3]),False,carla.LaneType.Driving)
                       is not None for p in world])


def draw(folder,lag,row,d,points,tags,meta,target):
    plt.rcParams.update({"font.family":"DejaVu Sans","text.color":"#e8eff6","axes.labelcolor":"#c7d5e2",
                         "xtick.color":"#aabed2","ytick.color":"#aabed2","font.size":10})
    ipm=np.load(folder/f"ipm-{row['frame']}.npy").transpose(1,2,0)
    h,count,intensity=d["lidar"].astype(float)/255.
    ground=d["plane"][0]*X[None,:]+d["plane"][1]*Y[:,None]+d["plane"][2]
    structure=np.clip((h*4.-.5-ground)/1.5,0.,1.)*(count>0)
    brightness=.07*count+.11*intensity+.48*structure
    base=np.clip(.85*ipm+brightness[...,None]*np.array([.8,.9,1.])+np.array([.027,.040,.060]),0,1)
    alpha=(.58*d["nav"].astype(float)/255.)[...,None]
    base=(1-alpha)*base+alpha*np.array([.10,.45,1.])
    fig,axes=plt.subplots(1,2,figsize=(9.1,7.9),gridspec_kw={"width_ratios":[.92,1.18]},facecolor="#101923")
    fig.subplots_adjust(left=.082,right=.98,top=.86,bottom=.19,wspace=.22)
    fig.text(.055,.95,f"Town04  /  {lag:.2f} s before pole contact",size=18,weight="bold")
    fig.text(.055,.918,f"Sensor {row['sensor_frame']}   |   Control {row['frame']}   |   {row['kmh']:.1f} km/h   |   raw 4.0 s plan",size=11)
    plan=np.vstack(([0.,0.],d["raw_xy"]))
    cen=np.array(meta["box_center"][:2]);l,w=meta["extent"][:2]
    corners=cen+np.array([[-l,-w],[l,-w],[l,w],[-l,w]])
    for ax,limits,title in zip(axes,[(25.6,-25.6,-16,112),(15,-16,-5,36)],["Full 128 × 128 input","Near field"]):
        ax.set_facecolor("#111c26")
        ax.imshow(base.transpose(1,0,2),origin="lower",extent=(-25.6,25.6,-16,112),interpolation="nearest")
        for match,col in [(tags!=meta["tags"]["Poles"],"#ff9d38"),(tags==meta["tags"]["Poles"],"#ffe380")]:
            pts=points[match]
            if len(pts):ax.scatter(pts[:,1],pts[:,0],s=2 if ax is axes[0] else 5,c=col,linewidths=0,alpha=.82)
        ax.plot(plan[:,1],plan[:,0],color="#64fa86",lw=2.1)
        ax.scatter(plan[10::10,1],plan[10::10,0],s=10,color="#adffbc")
        ax.add_patch(Polygon(corners[:,::-1],closed=True,ec="white",fc="none",lw=1.5))
        ax.annotate("",xy=(0,2.1),xytext=(0,-1.7),arrowprops={"arrowstyle":"->","color":"white","lw":1.4})
        ax.scatter([target[1]],[target[0]],c="#ff3856",edgecolors="white",s=48,zorder=6,linewidths=.75)
        ax.axis(limits);ax.set_aspect("equal");ax.set_title(title,color="white",fontsize=12)
        ax.set_xlabel("Lateral y (m)   +left");ax.set_ylabel("Forward x (m)")
        for spine in ax.spines.values():spine.set_color("#43536a")
    labels=[("Navigation Ch 6","#2879ff"),("Raw plan","#64fa86"),("Pursuit target","#ff3856"),
            ("Ego / heading","#ffffff"),("Wall / fence / rail","#ff9d38"),("Pole returns","#ffe380")]
    handles=[Line2D([],[],color=c,lw=3 if i in (0,1,3) else 0,marker="o" if i in (2,4,5) else None)
             for i,(_,c) in enumerate(labels)]
    fig.legend(handles,[s for s,_ in labels],loc="lower center",bbox_to_anchor=(.5,.055),ncol=3,
               frameon=False,labelcolor="#e8eff6",fontsize=10)
    fig.text(.055,.025,"Ground IPM + LiDAR height, density and intensity. Orange and gold show measured static surfaces.",fontsize=9)
    name=f"bev-minus-{lag:.1f}s.png"
    fig.savefig(folder/name,dpi=160,facecolor=fig.get_facecolor());plt.close(fig)
    return name


def export(folder):
    meta=json.loads((folder/"capture.json").read_bytes())
    road=carla.Map("Town04",(ROOT/".cache/v6-dual-system/carla-client/0.9.15/Carla/Maps/OpenDrive/Town04.xodr").read_text())
    crash=meta["crash_frame"]
    if any(not (folder/f"ipm-{crash-n}.npy").exists() for n in (40,20,6)):
        # Rendering uses the CARLA environment. Rebuild camera channels with the
        # production projection on the inference GPU in its own Python environment.
        subprocess.run([str(ROOT/".venv-train/Scripts/python.exe"),"-c",
            "import sys; from pathlib import Path; import numpy as np, torch\n"
            "from v6_dual_system.fast_model import GroundProjection\n"
            "p=Path(sys.argv[1]); ipm=GroundProjection().cuda().eval()\n"
            "with torch.inference_mode():\n"
            " for f in sys.argv[2:]:\n"
            "  with np.load(p/f'frame-{f}.npz') as d:\n"
            "   rgb,plane=(torch.as_tensor(d[k],device='cuda')[None] for k in ('rgb','plane'))\n"
            "   np.save(p/f'ipm-{f}.npy',ipm(rgb,plane)[0].cpu().numpy())",
            str(folder.resolve()),*(str(crash-n) for n in (40,20,6))],cwd=ROOT,check=True)
    commands={r["frame"]:r for r in map(json.loads,(folder/"commands.jsonl").read_bytes().splitlines())}
    rows=[];details={}
    cen=np.asarray(meta["box_center"][:2]);extent=np.asarray(meta["extent"][:2])
    for frame,row in commands.items():
        lag=(crash-frame)*meta["dt"]
        if lag>2.5:continue
        with np.load(folder/f"frame-{frame}.npz") as d:
            plan=d["raw_xy"]
            nav=ridge(d["nav"],d["route"])
            _,k=plan_curvature(plan)
            pts,tags=aligned_returns(d,meta)
            poles=tags==meta["tags"]["Poles"]
            rot=np.deg2rad(meta["box_yaw_deg"])
            metric=dict(t_to_crash=round(-lag,2),speed_kmh=row["kmh"],target_lane_shift=row["target_lane_shift"],
                nav_y_at_20m=first_crossing(nav,20.),pred_y_at_20m=first_crossing(plan,20.),
                max_traj_kappa=float(max(abs(k))) if len(k) else None,speed_limit_kmh=row["target_mps"]*3.6,
                steer_cmd=row["steer"],throttle=row["throttle"],brake=row["brake"],
                min_lidar_wall_dist_m=clearance(pts[~poles],cen,rot,extent),min_lidar_pole_dist_m=clearance(pts[poles],cen,rot,extent),
                curve_cap_kmh=row["curve_cap_mps"]*3.6,frame=frame,sensor_frame=row["sensor_frame"],
                control_x_target=row["pursuit_point_flu"][0],control_y_target=row["pursuit_point_flu"][1])
            rows.append(metric)
            if crash-frame not in (40,20,6):continue
            # Transform the actual control-frame pursuit point into the sensor frame for drawing.
            x,y=row["pursuit_point_flu"]
            loc=np.linalg.solve(d["sensor_pose"],d["control_pose"])@np.array([x,-y,0.,1.])
            target=loc[:2]*[1,-1]
            valid=driving(plan,d,road)
            navmask=driving(nav,d,road)
            core=np.argwhere((d["nav"]>=155)&(X[None,:]>=0))
            corepts=np.column_stack((X[core[:,1]],Y[core[:,0]]))
            navcore=driving(corepts,d,road)
            yaw=np.arctan2(*np.diff(np.vstack(([0.,0.],plan)),axis=0).T[::-1])
            proximity=[]
            for centre,heading in zip(plan,yaw):
                c,s=np.cos(heading),np.sin(heading)
                offset=cen@np.array([[c,s],[-s,c]])
                proximity.append(clearance(pts[poles],centre+offset,heading+rot,extent))
            near=nav[:,0]<=np.max(plan[:,0])
            info=dict(**metric,source_age_ms=row["source_age_ms"],target_sensor_flu=target.tolist(),
                      nav_ridge_drivable=[int(navmask.sum()),len(nav)],near_nav_ridge_drivable=[int(navmask[near].sum()),int(near.sum())],
                      nav_one_sigma_drivable=[int(navcore.sum()),len(navcore)],
                      plan_drivable=[int(valid.sum()),len(plan)],off_drivable_steps=(np.flatnonzero(~valid)+1).tolist(),
                      plan_footprint_over_pole_steps=[i+1 for i,q in enumerate(proximity) if q is not None and q<.001])
            info["image"]=draw(folder,lag,row,d,pts,tags,meta,target)
            details[str(round(lag,1))]=info
    for filename,table in (("all-frames.csv",rows),("quarter-second.csv",[r for r in rows if (crash-r["frame"])%5==0])):
        with (folder/filename).open("w",newline="",encoding="utf-8-sig") as out:
            writer=csv.DictWriter(out,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(table)
    (folder/"diagnosis.json").write_text(json.dumps(dict(meta=meta,snapshots=details),indent=2))
    print(json.dumps(details))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("folder",type=Path)
    export(parser.parse_args().folder)
