"""Cross-check logged FLU poses against independent velocity and camera handedness."""
import io
import json
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from .prepare_real_road_replay import OUT, SOURCE, ROOT, MetadataReader, write


def main():
    rows=json.loads((OUT/"cohort.json").read_text())
    routes=defaultdict(list)
    for row in rows:
        routes[row["route"]].append(row)
    checks=[]
    with tarfile.open(SOURCE/"openscene_metadata_mini.tgz","r|gz") as archive:
        for entry in archive:
            route=Path(entry.name).stem
            if route not in routes:continue
            frames=sorted(MetadataReader(io.BytesIO(archive.extractfile(entry).read())).load(),key=lambda x:x["timestamp"])
            for r in routes[route]:
                i=r["frame_index"]
                seq=frames[i:i+12]
                assert seq[0]["token"]==r["token"]
                ts=(np.asarray([f["timestamp"] for f in seq])-seq[0]["timestamp"])/1e6
                rotations=np.array([f["ego2global"][:3,:3] for f in seq])
                velocities=np.array([[*f["ego_dynamic_state"][:2],0.] for f in seq])
                local=(rotations[0].T @ rotations @ velocities[...,None])[:,:2,0]
                xy=np.cumsum((local[1:]+local[:-1])*.5*np.diff(ts)[:,None],axis=0)
                error=[np.interp(5,ts[1:],coord)-ref for coord,ref in zip(xy.T,r["truth_xy"][-1])]
                actor=json.loads((ROOT/r["actor_path"]).read_text())
                camera=actor["camera_calibration"]["CAM_FRONT"]
                transform=np.array(camera["world2cam"])@actor["world_from_ego"]
                rays=transform@np.array([[20,20],[1,-1],[0,0],[1,1]])
                # Column zero is ego-right in the CARLA-style projection API.
                assert rays[1,0]/rays[0,0]>rays[1,1]/rays[0,1] and min(rays[0])>0
                checks.append(dict(token=r["token"],group=r["group"],velocity_integration_error_xy_m=error))
    assert len(checks)==len(rows)
    errors=np.array([r["velocity_integration_error_xy_m"] for r in checks])
    norms=np.linalg.norm(errors,axis=1)
    write(OUT/"coordinate-check.json",dict(windows=len(rows),front_pinhole_sign_pass=len(checks),
        velocity_5s_error_median_m=float(np.median(norms)),velocity_5s_error_p95_m=float(np.percentile(norms,95)),
        signed_mean_xy_m=errors.mean(axis=0).tolist(),windows_detail=checks))


if __name__=="__main__":
    main()

