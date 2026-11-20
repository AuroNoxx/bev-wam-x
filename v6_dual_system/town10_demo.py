"""Scenario setup for the packaged UE5 Town10, without map input to the driver."""
import random

from .collect_fast_data import neighbor, path_from, wrap


def city_route(road_map, seed,long_demo=False):
    starts=list(road_map.get_spawn_points())
    if not starts:
        starts=[wp.transform for wp in road_map.generate_waypoints(30.) if not wp.is_junction]
    random.Random(seed).shuffle(starts)
    best=None
    for transform in starts[:96] if long_demo else starts:
        start=road_map.get_waypoint(transform.location)
        path=path_from(start,distance=230)
        if len(path)<=65 or any(w.is_junction for w in path[:(6 if long_demo else 24)]):continue
        if not long_demo:return path
        # At a fork prefer a fresh stretch of road, then the gentler turn.
        path=[start]
        visits={}
        for _ in range(2100):
            w=path[-1]
            key=(w.road_id,w.lane_id,int(w.s//10))
            visits[key]=visits.get(key,0)+1
            options=w.next(2.)
            if not options:break
            def rank(p):
                ahead=p.next(16.) or [p]
                return (min(visits.get((q.road_id,q.lane_id,int(q.s//10)),0) for q in ahead),
                        abs(wrap(p.transform.rotation.yaw-w.transform.rotation.yaw)),p.id)
            path.append(min(options,key=rank) if len(options)>1 else options[0])
        score=(len(path),len(visits))
        if best is None or score>best[0]:best=(score,path)
    if best:return best[1]
    raise RuntimeError("No suitable driving route on this map")


def add_traffic(world, road_map, route, manager, actors, seed, highway=False,highway_cruise=False,count=None):
    if count is None:count=10 if highway_cruise else 18 if highway else 8
    if count==0:return []
    library=world.get_blueprint_library()
    choices=[bp for bp in library.filter("vehicle.*") if
             any(k in bp.id for k in ("lincoln","dodge","mini","nissan","ford"))]
    rng=random.Random(seed)
    origin=route[0].transform.location
    starts=[tf for tf in road_map.get_spawn_points() if 12<tf.location.distance(origin)<160]
    rng.shuffle(starts)
    if count>len(starts):
        distant=[tf for tf in road_map.get_spawn_points() if tf.location.distance(origin)>=160]
        rng.shuffle(distant)
        starts+=distant
    if highway:
        # Populate lanes visible from the chase camera, continuing down the highway.
        nearby=[]
        indices=(range(16,min(500,len(route)),34) if highway_cruise else range(10,min(310,len(route)),12))
        for index in indices:
            wp=route[index+rng.randrange(min(5,len(route)-index))]
            lanes=[w for w in (neighbor(wp,True),wp,neighbor(wp,False)) if w]
            if highway_cruise and index==16:
                lanes=[w for w in lanes if w.id!=wp.id]
            for lane in rng.sample(lanes,min(len(lanes),1 if highway_cruise else rng.randint(1,2))):
                tf=lane.transform
                tf.location.z+=.45
                nearby.append(tf)
        starts=nearby+starts
    spawned=[]
    for tf in starts:
        other=world.try_spawn_actor(rng.choice(choices),tf)
        if other is None:continue
        actors.append(other)
        other.set_autopilot(True,manager.get_port())
        spawned.append(other.type_id)
        if len(spawned)==count:break
    return spawned
