"""Offline diagnostics on the final SFT checkpoint, with no optimizer or cache writes for edited prompts."""
from collections import defaultdict, Counter
import json
import argparse
import numpy as np
from .runtime import configure, ROOT, math_attention

configure()
import torch
from .real_dataset import RealData, inputs_for
from .train_disk_cached import DiskCachedTask
from .native_scene import encode_native_scene
from .kinematics import normalize_controls,denormalize_controls,integrate
from .data import intent_labels


def route_pairs(records):
    groups = defaultdict(lambda: defaultdict(list))
    counts = Counter()
    for r in records:
        if r['speed_kmh'] > 3:
            continue
        actor = json.loads((ROOT/r['actor_path']).read_text())
        if actor['target_speed_kmh'] != 0:
            continue
        with np.load(ROOT/r['label_path']) as label:
            d = np.linalg.norm(label['trajectory'],axis=-1).max()
        group = 'hold' if d < .5 else 'launch' if d >= 3 else 'intermediate'
        counts[group] += 1
        groups[r['route']][group].append(r)
    pairs = []
    for route, cases in sorted(groups.items()):
        choices = [(a,b) for a in cases['hold'] for b in cases['launch']]
        if choices:
            pairs.append(min(choices,key=lambda pair: (abs(pair[0]['speed_kmh']-pair[1]['speed_kmh']),
                                                        pair[0]['index'],pair[1]['index'])))
    return pairs,counts


def stop_probe(task,data,folder):
    pairs,counts = route_pairs(data.records)
    print('matched route pairs',len(pairs),'training zero-command counts',counts,flush=True)
    rows = []
    tokenizer = task.model.processor.tokenizer
    source = tokenizer.encode('0',add_special_tokens=False)
    replacement = tokenizer.encode('8',add_special_tokens=False)
    assert len(source)==len(replacement)==1
    for pair in pairs:
        for group,r in zip(('hold','launch'),pair):
            inputs = inputs_for(task.model.processor,r['actor_path'],task.device,task.token_ids)
            ids = inputs['input_ids']
            tokens = ids[0].tolist()
            marker = 'target cruise: 0.00 km/h'
            original = tokenizer.decode(tokens)
            assert original.count(marker)==1
            candidates = [i for i,t in enumerate(tokens) if t==source[0]]
            expected = original.replace(marker,'target cruise: 8.00 km/h')
            pos, = [i for i in candidates if tokenizer.decode(tokens[:i]+replacement+tokens[i+1:])==expected]
            with np.load(ROOT/r['label_path']) as label:
                speed=torch.as_tensor(label['v0'],device=task.device).reshape(1)
                true_x=float(label['trajectory'][-1,0])
                expert_ax=float(label['controls'][:,0].mean())
            for target in (0,8):
                changed = ids.clone()
                if target:
                    changed[0,pos]=replacement[0]
                inp = {**inputs,'input_ids':changed}
                native=encode_native_scene(task.model,inp)
                context=task.planner.context(native,inp)
                for seed in (1907,1908,1909,1910):
                    torch.manual_seed(seed)
                    noise=torch.randn(1,50,2,device=task.device)
                    out=task.planner.sample(context,speed,noise,steps=10)
                    tr=out['trajectory'][0]
                    rows.append(dict(group=group,route=r['route'],index=r['index'],v0_kmh=r['speed_kmh'],
                                     seed=seed,target=target,true_x=true_x,expert_ax=expert_ax,
                                     x5=float(tr[-1,0]),v5=float(tr[-1,3]),mean_ax=float(tr[:,4].mean()),
                                     ax_first=float(tr[:10,4].mean()),ax_last=float(tr[-10:,4].mean()),
                                     speed_floor_contribution=float(tr[-1,3]-speed.item()-.1*tr[:,4].sum())))
        print('route pair measured',pair[0]['route'],flush=True)
    (folder/'stop-probe.json').write_text(json.dumps(dict(zero_target_training_counts=dict(counts),rows=rows),indent=2))


def world_probe(task,data,folder):
    rows=[]
    bins=512
    hist=np.zeros((3,2,2,bins),dtype=np.int64)
    for i,batch in enumerate(data.val_dataloader()):
        r=batch['record']
        labels={k:v.cuda() for k,v in batch['labels'].items()}
        native,_=task.encode_path(r['actor_path'])
        current=task.world.encode_current(native['hidden'])
        xy,v=labels['trajectory'],labels['v0']
        pred=task.world(current,xy,v)['occupancy'][0]
        static=task.world.decoder(current).float().sigmoid()[0,0]
        # A smooth lateral displacement after t=1 s, solely to probe suffix dependence.
        alt=xy.clone()
        phase=(torch.arange(1,51,device=v.device)*.1-1).clamp(0,4)/4
        alt[:,:,1]+=3.5*(3*phase.square()-2*phase.pow(3))
        assert torch.equal(alt[:,:10],xy[:,:10])
        diff=(task.world(current,alt,v)['occupancy'][0,0]-pred[0]).abs()
        for h in range(3):
            truth=labels['occupancy'][0,h].float()
            now=labels['current_occupancy'][0].float()
            known=labels['support'][0,h].bool() & labels['current_support'][0].bool()
            changed=known & (truth!=now)
            row=dict(route=r['route'],index=r['index'],h=h+1,valid=int(known.sum()),changed=int(changed.sum()))
            for name,field in [('world',pred[h]),('static',static)]:
                error=(field-truth).square()
                row[name+'_brier']=float(error[known].mean())
                row[name+'_changed_sse']=float(error[changed].sum())
                for event,mask in [('appeared',changed & (truth>0)),('cleared',changed & (truth==0))]:
                    row[event]=int(mask.sum())
                    row[name+'_'+event+'_sse']=float(error[mask].sum())
                for state,mask in [('stayed_free',known & (truth==0) & (now==0)),
                                   ('stayed_occupied',known & (truth>0) & (now>0))]:
                    row[state]=int(mask.sum())
                    row[name+'_'+state+'_sse']=float(error[mask].sum())
                # Fixed FLU forward corridor, x in (0,112) m and |y| <= 2 m.
                x=torch.arange(128,device=v.device)+.5-16
                y=(torch.arange(128,device=v.device)+.5)*.3-19.2
                corridor=known & (x[None,:]>0) & (y[:,None].abs()<=2)
                near=corridor & (x[None,:]<80)
                score=field[near].float().cpu().numpy()
                occupied=truth[near].bool().cpu().numpy()
                indices=np.minimum((score*bins).astype(int),bins-1)
                for cls in (0,1):
                    hist[h,int(name=='static'),cls]+=np.bincount(indices[occupied==cls],minlength=bins)
                for region,mask in [('ahead_0_40',corridor & (x[None,:]<40)),
                                    ('ahead_40_80',corridor & (x[None,:]>=40) & (x[None,:]<80)),
                                    ('ahead_80_112',corridor & (x[None,:]>=80))]:
                    positive=mask & (truth>0)
                    row[region+'_valid']=int(mask.sum())
                    row[region+'_positives']=int(positive.sum())
                    row[name+'_'+region+'_tp']=int((positive & (field>=.5)).sum())
                    row[name+'_'+region+'_predicted']=int((mask & (field>=.5)).sum())
            if h==0:
                row.update(suffix_mean=float(diff[known].mean()),suffix_max=float(diff[known].max()))
            rows.append(row)
    (folder/'world-probe.json').write_text(json.dumps(rows,indent=2))
    np.savez_compressed(folder/'front-corridor-hist.npz',counts=hist,bins=bins)


def solver_probe(task,data,folder):
    pairs,_ = route_pairs(data.records)
    rows = []
    for pair in pairs:
        for group,r in zip(('hold','launch'),pair):
            native,inp = task.encode_path(r['actor_path'])
            ctx = task.planner.context(native,inp)
            with np.load(ROOT/r['label_path']) as label:
                v=torch.as_tensor(label['v0'],device=task.device).reshape(1)
                ctrl=torch.as_tensor(label['controls'],device=task.device)[None]
                truth=torch.as_tensor(label['trajectory'],device=task.device)[None]
            clean=normalize_controls(ctrl)
            reason=int(task.reason_head(native['hidden'][:,-2]).argmax())
            for seed in (1907,1908,1909,1910):
                torch.manual_seed(seed)
                noise=torch.randn_like(clean)
                row=dict(group=group,route=r['route'],index=r['index'],seed=seed,
                         intent_true=int(intent_labels(ctrl,truth)),intent_pred=reason)
                for steps in (10,40):
                    traj=task.planner.sample(ctx,v,noise,steps=steps)['trajectory']
                    row[f'x{steps}']=float(traj[0,-1,0])
                    row[f'ax{steps}']=float(traj[...,4].mean())
                for t in (0,.25,.5,.75,.9):
                    mixed=noise*(1-t)+t*clean
                    field=task.planner.velocity(mixed,noise.new_tensor([t]),ctx)
                    estimate=denormalize_controls(mixed+(1-t)*field)
                    row[f'x_oracle_{t}']=float(integrate(estimate,v)[0,-1,0])
                rows.append(row)
    (folder/'solver-probe.json').write_text(json.dumps(rows,indent=2))


def cohort_probe(task,data,folder):
    rows=[]
    for split,records in [('train',data.records),('validation',data.validation)]:
        for r in records:
            if r['speed_kmh']>3:
                continue
            actor=json.loads((ROOT/r['actor_path']).read_text())
            if actor['target_speed_kmh']!=0:
                continue
            with np.load(ROOT/r['label_path']) as label:
                xy=torch.as_tensor(label['trajectory'],device=task.device)[None]
                ctrl=torch.as_tensor(label['controls'],device=task.device)[None]
                v=torch.as_tensor(label['v0'],device=task.device).reshape(1)
            distance=float(xy.norm(dim=-1).max())
            group='hold' if distance<.5 else 'launch' if distance>=3 else 'intermediate'
            native,inp=task.encode_path(r['actor_path'])
            prob=task.reason_head(native['hidden'][:,-2]).float().softmax(-1)
            ctx=task.planner.context(native,inp)
            torch.manual_seed(1907)
            pred=task.planner.sample(ctx,v,torch.randn(1,50,2,device=task.device),steps=10)['trajectory']
            rows.append(dict(split=split,group=group,route=r['route'],index=r['index'],
                             label=int(intent_labels(ctrl,xy)),prediction=int(prob.argmax()),
                             p_accelerate=float(prob[0,9:12].sum()),x5=float(pred[0,-1,0]),
                             true_x=float(xy[0,-1,0]),v0_kmh=r['speed_kmh']))
        print('cohort completed',split,flush=True)
    (folder/'stop-cohort.json').write_text(json.dumps(rows,indent=2))


def scene_probe(task,data,folder):
    cases=[]
    for batch in data.val_dataloader():
        r=batch['record']
        native,_=task.encode_path(r['actor_path'])
        latent=task.world.encode_current(native['hidden']).cpu()
        cases.append((r,batch['labels'],latent))
    cases.sort(key=lambda c:(c[0]['route'],c[0]['index']))
    rows=[]
    for i,(r,labels,latent) in enumerate(cases):
        donor,_,swapped=cases[(i+len(cases)//2)%len(cases)]
        assert donor['route']!=r['route']
        xy=labels['trajectory'].cuda()
        speed=labels['v0'].cuda()
        for condition,feature in [('observed',latent),('other_route',swapped)]:
            current=feature.cuda()
            p0=task.world.decoder(current).float().sigmoid()[0,0]
            future=task.world(current,xy,speed)['occupancy'][0]
            for h,field in enumerate([p0,*future]):
                if h:
                    truth=labels['occupancy'][0,h-1].cuda().bool()
                    known=(labels['support'][0,h-1]&labels['current_support'][0]).cuda().bool()
                else:
                    truth=labels['current_occupancy'][0].cuda().bool()
                    known=labels['current_support'][0].cuda().bool()
                row=dict(route=r['route'],index=r['index'],donor_route=donor['route'],donor_idx=donor['index'],
                         condition=condition,h=h)
                error=(field-truth.float()).square()
                row['brier']=float(error[known].mean())
                row['occupied']=int((truth & known).sum())
                row['occupied_sse']=float(error[truth & known].sum())
                row['free']=int((~truth & known).sum())
                row['free_sse']=float(error[~truth & known].sum())
                rows.append(row)
    (folder/'scene-probe.json').write_text(json.dumps(rows,indent=2))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--solver',action='store_true')
    parser.add_argument('--cohort',action='store_true')
    parser.add_argument('--world-only',action='store_true')
    parser.add_argument('--scene',action='store_true')
    args=parser.parse_args()
    folder=ROOT/'artifacts/v5-qwen-drive-world/subset2000'
    data=RealData('subset2000')
    torch.manual_seed(71)
    torch.set_num_threads(6)
    torch.cuda.set_per_process_memory_fraction(.8)
    ckpt=torch.load(folder/'latest.ckpt',map_location='cpu',weights_only=False,mmap=True)
    assert ckpt['global_step']==750
    task=DiskCachedTask(ckpt['task_state']['config'],data.identity,'subset2000').cuda()
    task.on_load_checkpoint(ckpt)
    task.load_state_dict(ckpt['state_dict'])
    del ckpt
    task.eval()
    with torch.no_grad(),math_attention(),torch.autocast('cuda',dtype=torch.bfloat16):
        if args.scene:
            scene_probe(task,data,folder/'final-window-audit')
        elif args.world_only:
            world_probe(task,data,folder/'final-window-audit')
        elif args.cohort:
            cohort_probe(task,data,folder/'final-window-audit')
        elif args.solver:
            solver_probe(task,data,folder/'final-window-audit')
        else:
            stop_probe(task,data,folder/'final-window-audit')
            world_probe(task,data,folder/'final-window-audit')
    print('completed',flush=True)


if __name__=='__main__':
    main()
