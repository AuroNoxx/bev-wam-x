"""Report matched training, development and validation planning errors."""
import json

import numpy as np
from .geometry_repair_features import BASE, RUN, ROOT, RealData, split_records, torch, math_attention
from .kinematics import integrate
from .motion_profile import profile_features
from .paired_launch_training import samples as paired_samples
from .mine_launch_pairs import PAIR, SOURCE, load
from .train_scale_profile import ScaleMotion, extend_observation, fingerprint
from .replay_sft_training_set import OUT, commit


def stats(rows, names):
    return dict(n=len(rows),routes=len(set(r["route"] for r in rows)),
                better_than_cv=sum(r["ade"]<r["cv_ade"] for r in rows),ade_below_02=sum(r["ade"]<.2 for r in rows),
                means={k:float(np.mean([r[k] for r in rows])) for k in names}) if rows else dict(n=0)


def head_replay(data):
    saved=load(PAIR,"validation.json")
    sha=fingerprint(PAIR/"best.ckpt")
    assert sha==saved["checkpoint_sha256"]
    ck=torch.load(PAIR/"best.ckpt",map_location="cpu",weights_only=False)
    ident=ck["task_state"]["identity"]
    assert ck["global_step"]==162 and ident["manifest"]==data.identity
    assert ident["observations"]==fingerprint(SOURCE/"seeded/observations.json")
    assert ident["pair_manifest"]==fingerprint(PAIR/"cohort.json")
    assert ident["pair_observations"]==fingerprint(PAIR/"pair-observations.json")
    plan=ScaleMotion(ident).cuda().eval()
    plan.load_state_dict(ck["state_dict"])
    cues=load(SOURCE/"seeded","observations.json")["splits"]
    def measure(r,obs,base,raw,v,truth):
        obs,base,raw,v=(x.cuda() for x in (obs,base,raw,v))
        _,tr,_=plan.profile(obs,base,raw,v.flatten())
        old=integrate(raw,v.flatten())[0,:,:2].float().cpu()
        cv=torch.stack([torch.arange(1,51)*.1*float(v.item()),torch.zeros(50)],1)
        paths=torch.stack([tr[0,:,:2].float().cpu(),old,cv])
        err=(paths-truth).norm(dim=2)
        return dict(route=r["route"],actor_path=r["actor_path"],ade=float(err[0].mean()),
            fde5=float(err[0,-1]),base_ade=float(err[1].mean()),cv_ade=float(err[2].mean()))
    groups={}
    records=split_records(data)
    for split,indices in [("fit",ident["indices"]["fit"]),("development",ident["indices"]["development"]),
                          ("validation",range(len(data.validation)))]:
        group="validation" if split=="validation" else "train"
        rows=[]
        for i in indices:
            r=records[group][i]
            feat=torch.load(RUN/f"features/{group}/{i:04d}.pt",weights_only=True)
            cue=cues[group][i]
            assert feat.pop("actor_path")==cue["actor_path"]==r["actor_path"]
            f={k:v[None].cuda() for k,v in feat.items()}
            obs,base=profile_features(f)
            with np.load(ROOT/r["label_path"]) as lab:truth=torch.from_numpy(lab["trajectory"])
            row=measure(r,extend_observation(obs[0],cue)[None],base,f["ctrl"],f["v0"],truth)
            if split=="validation":assert abs(row["ade"]-saved["rows"][i]["ade"])<1e-4
            rows.append(row)
        groups[split]=rows
    groups["extra_fit"]=[]
    for r,s in paired_samples()["fit"]:
        obs,base,raw,v,truth=s[:5]
        groups["extra_fit"].append(measure(r,obs[None],base[None],raw[None],v[None],truth))
    fit=groups["fit"]+groups["extra_fit"]
    assert len({r["actor_path"] for r in fit})==len(fit)
    summary={g:stats(rr,["ade","fde5","base_ade","cv_ade"]) for g,rr in dict(**groups,all_fit=fit).items()}
    commit(OUT/"head-windows.json",dict(sha256=sha,groups=groups,summary=summary))
    return summary


def main():
    torch.set_num_threads(3)
    data=RealData("subset2000")
    with torch.inference_mode(),math_attention(),torch.autocast("cuda",dtype=torch.bfloat16):
        head=head_replay(data)
    result=load(OUT,"windows.json")
    windows=list(result["rows"].values())
    assert len(windows)==2000 and sum(r["draws"] for r in windows)==6000
    canonical=np.load(BASE/"final-window-audit/predictions.npz")
    scored=json.loads((BASE/"validation.jsonl").read_text().splitlines()[-1])
    assert scored["step"]==750
    val=[]
    for i,r in enumerate(data.validation):
        assert canonical["route"][i]==r["route"] and canonical["index"][i]==r["index"]
        with np.load(ROOT/r["label_path"]) as lab:
            truth=lab["trajectory"]
            v=float(lab["v0"])
            lab_xy=integrate(torch.from_numpy(lab["controls"][None]),torch.tensor([v]))[0,:,:2].numpy()
        e=np.linalg.norm(canonical["trajectories"][i,:,:2]-truth,axis=1)
        assert abs(e.mean()-scored["samples"][i]["ade_m"])<1e-4
        cv=np.c_[np.arange(1,51)*.1*v,np.zeros(50)]
        val.append(dict(ade=float(e.mean()),fde5=float(e[-1]),cv_ade=float(np.linalg.norm(cv-truth,axis=1).mean()),
                  label_ade=float(np.linalg.norm(lab_xy-truth,axis=1).mean()),speed=r["speed_kmh"],source=r["source"],route=r["route"]))
    groups={"all_train":windows,"validation":val,
        "subset2000_drawn":[r for r in windows if r["draws"]],"subset2000_not_drawn":[r for r in windows if not r["draws"]],
        "overfit64":[r for r in windows if r["overfit64"]],"seen_either_phase":[r for r in windows if r["draws"] or r["overfit64"]],
        "seen_and_label_ade_lt_0.2":[r for r in windows if r["draws"] and r["label_ade"]<.2],
        "validation_label_ade_lt_0.2":[r for r in val if r["label_ade"]<.2]}
    for split,rows in [("train_drawn",groups["subset2000_drawn"]),("validation",val)]:
        for b in range(3):
            groups[f"{split}_speed_{b*40}"]=[r for r in rows if min(2,int(r["speed"]//40))==b]
    names=["ade","fde5","cv_ade","label_ade"]
    base={g:stats(rr,names) for g,rr in groups.items()}
    # Compare on the validation source/speed mixture, retaining independent routes.
    strata={}
    for r in val:
        key=(r["source"],int(r["speed"]//20))
        strata[key]=strata.get(key,0)+1
    matched={k:0. for k in names}
    support=[]
    for (source,bucket),n in sorted(strata.items()):
        rr=[r for r in groups["subset2000_drawn"] if r["source"]==source and int(r["speed"]//20)==bucket]
        assert rr
        support.append(dict(source=source,lower_kmh=bucket*20,train_windows=len(rr),validation_windows=n))
        for k in names:matched[k]+=float(np.mean([r[k] for r in rr]))*n/len(val)
    matches=list(result["rows"][r["actor_path"]] for r in data.manifest["overfit64"])
    previous=json.loads((BASE.parent/"overfit64/validation.jsonl").read_text().splitlines()[-1])
    assert previous["epoch"]==30
    paired64={}
    for label,rr in [("all",list(zip(matches,previous["samples"],strict=True))),
                     ("good_labels",[(a,b) for a,b in zip(matches,previous["samples"],strict=True) if a["label_ade"]<.2])]:
        assert all(a["route"]==b["route"] and abs(a["speed"]-b["speed_kmh"])<.001 for a,b in rr)
        paired64[label]=dict(n=len(rr),before=float(np.mean([b["ade_m"] for _,b in rr])),
                       after=float(np.mean([a["ade"] for a,_ in rr])),improved=sum(a["ade"]<b["ade_m"] for a,b in rr))
    commit(OUT/"summary.json",dict(identity=result["identity"],step750=base,step162=head,paired64=paired64,
        train_weighted_to_validation=dict(support=support,means=matched)))
    print(json.dumps(dict(base=base,head=head,paired64=paired64)),flush=True)


if __name__=="__main__":
    main()

