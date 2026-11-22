"""Build sufficiently long, connected collection missions in the locked geographic split."""
import numpy as np
import carla

from .navigation import directed_step, waypoint_forward
from .scenarios import xyz, same_direction


def in_region(w,split,bounds):
    from .bulk import allowed
    return allowed(xyz(w)[:2],split,bounds)


def extend(path, distance, split, bounds, rng, backward=False):
    traveled=0.
    while traveled<distance:
        w=path[-1]
        opts=w.previous(2.) if backward else w.next(2.)
        opts=[p for p in opts if (directed_step(p,w) if backward else directed_step(w,p)) and in_region(p,split,bounds)]
        if not opts:
            break
        # Prefer geometric continuity. Mission diversity comes from different starting roads.
        p=max(opts,key=lambda p:waypoint_forward(w)@waypoint_forward(p)+.005*rng.random())
        delta=np.linalg.norm(xyz(p)[:2]-xyz(w)[:2])
        if delta<.15 or delta>6:
            break
        path.append(p)
        traveled+=delta
    return traveled


def build(road,mission,bounds,sampled,seed):
    rng=np.random.default_rng(seed)
    speed=mission.speed/3.6
    needed=125*speed+200
    split=mission.split
    if split=='val':
        # Reuse the surveyed exit branch at 794 -> 580. Walk backward from the
        # ramp to the requested approach, then forward through it.
        starts=[w for w in sampled if w.road_id==580 and in_region(w,split,bounds)]
        starts.sort(key=lambda w:np.linalg.norm(xyz(w)[:2]-[-2250,4900]))
        for trial in range(60):
            if not starts: break
            w=starts[trial%len(starts)]
            back=[w]
            if extend(back,540,split,bounds,rng,True)<440 or not any(p.road_id==794 for p in back):continue
            path=back[::-1]
            if extend(path,needed-520,split,bounds,rng)>speed*102-420:return path
        # Survey route already resolves ambiguous outgoing branches using the road sketch.
        from .bulk import expert_route
        path=expert_route(road,mission,bounds,sampled)
        if not mission.required_roads.issubset({p.road_id for p in path}):
            raise RuntimeError('Validation route fails surveyed highway/ramp topology')
        return path
    if 'interchange' in mission.name:
        centers=[]
        for w in sampled:
            if not w.is_junction or not in_region(w,split,bounds):continue
            n=w.next(18)
            if len(n)!=1:continue
            curvature=np.linalg.norm(waypoint_forward(w)[:2]-waypoint_forward(n[0])[:2])/18
            # Select a curved interchange adjoining a multilane highway nearby.
            if .009<curvature<.085:
                centers.append(w)
        rng.shuffle(centers)
        for w in centers[:1800]:
            pre=[w]
            if extend(pre,250,split,bounds,rng,True)<245:continue
            if not any(same_direction(p,'right') or same_direction(p,'left') for p in pre[::8]):continue
            path=pre[::-1]
            if extend(path,needed-245,split,bounds,rng)>=needed-250:return path
        raise RuntimeError('No eligible curved interchange route')
    direction=mission.delta[0]/mission.length[0]
    candidates=[w for w in sampled if np.linalg.norm(xyz(w)[:2]-mission.xy[0])<120 and
                waypoint_forward(w)[:2]@direction>.75 and not w.is_junction and in_region(w,split,bounds)]
    candidates.sort(key=lambda w:np.linalg.norm(xyz(w)[:2]-mission.xy[0])+rng.uniform(0,22))
    for w in candidates[:80]:
        path=[w]
        if extend(path,needed,split,bounds,rng)>=needed-5:return path
    raise RuntimeError(f'No connected route with future support for {mission.name}')
