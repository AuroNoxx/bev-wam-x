"""Collect matched real command branches and speed-diverse recoveries for stage 2."""
import json

from .collect_fast_data import episode
from .sensors import DATA, DT, local_client

REPLACE={"highway":560,"left":900,"right":900,"recovery":900}


def main():
    client=local_client(2176)
    world=client.get_world()
    if not world.get_map().name.endswith("/Town04"):world=client.load_world("Town04")
    road=world.get_map()
    original=world.get_settings()
    cfg=world.get_settings()
    cfg.synchronous_mode,cfg.fixed_delta_seconds,cfg.no_rendering_mode=True,DT,False
    root=DATA/"Town04-corrections"
    root.mkdir(exist_ok=True)
    world.apply_settings(cfg)
    try:
        for case in range(40):
            counts={s:sum(json.loads(p.read_bytes())["count"] for p in root.glob(s+"-*/complete.json")) for s in REPLACE}
            if all(counts[s]>=n for s,n in REPLACE.items()):return
            speed=(55,85,105,120)[case%4]
            event=100+case%4*42
            for branch,scene in enumerate(("highway","left","right","recovery")):
                # Keep all three counterfactuals of each paired initial state in Town04/train.
                if scene=="recovery" and counts[scene]>=REPLACE[scene]:continue
                if scene!="recovery" and all(counts[s]>=REPLACE[s] for s in ("highway","left","right")):continue
                seed=36000+case*4+branch
                folder=root/f"{scene}-{seed}"
                if folder.exists():continue
                plan=dict(route_seed=31000+case,speed=speed,event_tick=event,
                          ticks=event+190,route_kind="paired")
                if scene=="recovery":
                    plan.update(route_kind="recovery",ticks=1000,push_min_mps=.80*speed/3.6)
                count=episode(client,world,road,"Town04","train",scene,seed,2000,folder,plan)
                print(scene,seed,speed,count,flush=True)
        raise RuntimeError("Corrective collection did not fill its fixed replacement quotas")
    finally:
        world.apply_settings(original)


if __name__=="__main__":
    main()
