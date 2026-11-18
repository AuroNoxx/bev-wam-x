"""Prepare a deterministic real-road planning replay from OpenScene mini."""
import hashlib
import importlib
import io
import json
import pickle
import tarfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/public/openscene-mini-audit"
OUT = ROOT / "artifacts/v5-qwen-drive-world/real-road-mini"
RGB = ROOT / "data/v5-qwen-drive-world/real-road-mini"
SHARDS = {
    4: "c2bdd4cb3231a49a784941700471b1c6dd50b70fd82bce5c814349cfe9d8e2be",
    5: "74df953723507a8dcd778c6d1feb2ef2a3e57b799a6e3b35f1728cb72384a595",
    6: "ac4d3b9ea7d0e3c2584f716a841b867ce45e76699e5aefaa61016c1a083c3f7b",
    19: "fd7ab5c59c9cb0b216dac6d07f3f4cf81a5c9525ea942277fc789109ddc64359"
}
CAMS = ["CAM_F0","CAM_L0","CAM_R0","CAM_B0","CAM_L1","CAM_R1"]
NAMES = ["CAM_FRONT","CAM_FRONT_LEFT","CAM_FRONT_RIGHT","CAM_BACK","CAM_BACK_LEFT","CAM_BACK_RIGHT"]
VIEWS = ["<FRONT VIEW>","<FRONT LEFT VIEW>","<FRONT RIGHT VIEW>","<BACK VIEW>","<BACK LEFT VIEW>","<BACK RIGHT VIEW>"]
FLIP = np.diag([1,-1,1,1])
OPTICAL = np.array([[0,1,0,0],[0,0,-1,0],[1,0,0,0],[0,0,0,1]])
TIMES = np.arange(1,11) * .5


class MetadataReader(pickle.Unpickler):
    def find_class(self, module, name):
        if ((module == "numpy" and name in ("dtype","ndarray")) or
            (module in ("numpy.core.multiarray","numpy._core.multiarray") and name in ("_reconstruct","scalar")) or
            (module in ("numpy.core.numeric","numpy._core.numeric") and name == "_frombuffer")):
            return getattr(importlib.import_module(module), name)
        raise pickle.UnpicklingError(f"Unexpected metadata type {module}.{name}")


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")


def interp(times, xp, values):
    return np.stack([np.interp(times, xp, col) for col in values.T], axis=-1)


def camera(frame, name):
    cal = frame["cams"][name]
    optical = np.eye(4)
    optical[:3,:3] = cal["sensor2lidar_rotation"]
    optical[:3,3] = cal["sensor2lidar_translation"]
    mount = FLIP @ frame["lidar2ego"] @ optical @ OPTICAL
    pose = FLIP @ frame["ego2global"] @ FLIP
    return dict(world2cam=np.linalg.inv(pose @ mount).tolist(), intrinsic=cal["cam_intrinsic"].tolist(),
                image_size_x=1920,image_size_y=1080)


def windows(frames):
    frames.sort(key=lambda f:f["timestamp"])
    stamps = np.array([f["timestamp"] for f in frames]) / 1e6
    poses = np.stack([f["ego2global"] for f in frames])
    speed = np.array([np.linalg.norm(f["ego_dynamic_state"][:2]) for f in frames])
    candidates = []
    for i in range(4,len(frames)-11):
        sl = slice(i-4,i+12)
        t = stamps[sl]-stamps[i]
        if np.any(np.abs(np.diff(t)-.5)>.07) or max(abs(t[1:5]-np.arange(-1.5,.1,.5)))>.08:
            continue
        if len({f["scene_token"] for f in frames[sl]}) != 1:
            continue
        local = np.linalg.inv(poses[i]) @ poses[sl]
        yaw = np.unwrap(np.arctan2(local[:,1,0],local[:,0,0]))
        xy = interp(TIMES,t,local[:,:2,3])
        v = np.interp(TIMES,t,speed[sl])
        heading = abs(np.interp(5.,t,yaw))
        if speed[i]>=3 and speed[i]-v[-1]>=3 and heading < .18:
            kind = "brake"
        elif speed[i]>=2 and heading>=np.deg2rad(18) and np.linalg.norm(xy[-1])>=8:
            kind = "turn"
        elif speed[i]>=20/3.6 and np.max(abs(v-speed[i]))<=1 and heading<.08 and abs(xy[-1,1])<=1.5:
            kind = "cruise"
        else:
            continue
        priority = hashlib.sha256(("openscene-pilot-1/"+frames[i]["token"]).encode()).digest()
        candidates.append((priority,i,kind,xy,v))
    accepted = []
    counts = Counter()
    for _,i,kind,xy,v in sorted(candidates,key=lambda row:row[0]):
        if counts[kind]>=2 or any(abs(stamps[i]-stamps[j])<7 for j in accepted):
            continue
        counts[kind]+=1
        accepted.append(i)
        now = frames[i]
        route = now["log_name"]
        hist = frames[i-4:i+1]
        ts = stamps[i-4:i+1]-stamps[i]
        local = np.linalg.inv(poses[i]) @ poses[i-4:i+1]
        grid = np.arange(-15,1)*.1
        history = interp(grid,ts,np.c_[local[:,:2,3],np.unwrap(np.arctan2(local[:,1,0],local[:,0,0]))])
        dyn = np.array([f["ego_dynamic_state"] for f in hist])
        planar = local[:,:2,:2]
        velocity = interp(grid,ts,(planar @ dyn[:,:2,None]).squeeze(-1))
        acceleration = interp(grid,ts,(planar @ dyn[:,2:4,None]).squeeze(-1))
        imgs = {view:[(RGB/f["cams"][cam]["data_path"]).relative_to(ROOT).as_posix()
                      for f in (hist[-4:] if cam==CAMS[0] else [now])]
                for cam,view in zip(CAMS,VIEWS)}
        actor_path = (RGB/"actors"/(now["token"]+".json")).relative_to(ROOT)
        actor = dict(source="openscene",route=route, timestamp=float(stamps[i]),images=imgs,
                     history=history.tolist(),history_velocity=velocity.tolist(),history_acceleration=acceleration.tolist(),
                     ego_velocity=dyn[-1,:2].tolist(),ego_acceleration=dyn[-1,2:4].tolist(),speed_kmh=speed[i]*3.6,
                     driving_command=[0,0,0,0],nav_command=0,navigation_flu=[0.,0.],
                     target_speed_kmh=speed[i]*3.6, logged_goal_available=False,
                     world_from_ego=(FLIP@poses[i]@FLIP).tolist(),
                     camera_calibration={name:camera(now,cam) for name,cam in zip(NAMES,CAMS)})
        write(ROOT/actor_path,actor)
        yield dict(actor_path=actor_path.as_posix(),route=route,city=now["map_location"],group=kind,
                   speed_kmh=actor["speed_kmh"],paths=imgs[VIEWS[0]][-3:],ego=actor["world_from_ego"],
                   cameras=[camera(f,"CAM_F0") for f in hist[-3:]],
                   truth_xy=xy.tolist(),truth_speed=v.tolist(),frame_index=i,token=now["token"])


def main():
    archives = []
    logs = set()
    for shard, sha in SHARDS.items():
        path = SOURCE / f"camera_{shard}.tgz"
        download = path if path.exists() else path.with_suffix(".tgz.part")
        with download.open("rb") as stream:
            if hashlib.file_digest(stream,"sha256").hexdigest()!=sha:
                raise ValueError(f"Shard {shard} hash mismatch")
        if download != path:
            download.rename(path)
        archives.append(path)
        with tarfile.open(path,"r|gz") as arc:
            for member in arc:
                if member.name.endswith(".jpg"):
                    logs.add(Path(member.name).parts[-3])
    rows = []
    metadata = SOURCE/"openscene_metadata_mini.tgz"
    with metadata.open("rb") as stream:
        if hashlib.file_digest(stream,"sha256").hexdigest()!="c2e08867e062ba916eccd56df191e728d42fb94de854c28f2226eed14cecd8be":
            raise ValueError("OpenScene mini metadata hash mismatch")
    with tarfile.open(metadata,"r|gz") as arc:
        for member in arc:
            if Path(member.name).stem in logs and member.name.endswith(".pkl"):
                frames=MetadataReader(io.BytesIO(arc.extractfile(member).read())).load()
                rows.extend(windows(frames))
    if not rows:
        raise ValueError("No valid windows")
    wanted = set()
    for row in rows:
        actor = json.loads((ROOT/row["actor_path"]).read_text())
        wanted.update((ROOT/p).relative_to(RGB).as_posix() for group in actor["images"].values() for p in group)
    for path in archives:
        with tarfile.open(path,"r|gz") as arc:
            for member in arc:
                key="/".join(Path(member.name).parts[-3:])
                if key not in wanted:continue
                blob=arc.extractfile(member).read()
                with Image.open(io.BytesIO(blob)) as im:
                    assert im.size==(1920,1080) and im.format=="JPEG"
                    im.verify()
                dest=RGB/key
                dest.parent.mkdir(parents=True,exist_ok=True)
                dest.write_bytes(blob)
                wanted.remove(key)
    if wanted:raise ValueError(f"Missing {len(wanted)} images")
    write(OUT/"cohort.json",rows)
    print(len(logs),len(rows),dict(Counter(r["group"] for r in rows)),flush=True)


if __name__=="__main__":
    main()

