"""Offline multidimensional metrics. Missing is None, never a safety zero."""
import math,statistics,json,hashlib

def metric(value,unit,reason=None,n=0):
    return {'value':value,'unit':unit,'available':value is not None,'reason':reason,'n':n}

def finite(x):return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)

def percentile(xs,q):
    xs=sorted(xs);a=(len(xs)-1)*q;lo=int(a);hi=min(lo+1,len(xs)-1);return xs[lo]*(hi-a)+xs[hi]*(a-lo) if hi!=lo else xs[lo]

def derivatives(rows):
    # Do not bridge missing samples; reject non-monotonic timestamps.
    ts=[r.get('t_s') for r in rows]
    if any(not finite(t) for t in ts) or any(b<=a for a,b in zip(ts,ts[1:])):raise ValueError('strict finite simulation timestamps required')
    accel=[];jerk=[]
    for a,b in zip(rows,rows[1:]):
        dt=b['t_s']-a['t_s']
        if finite(a.get('speed_mps')) and finite(b.get('speed_mps')) and dt<=.25:
            accel.append(((a['t_s']+b['t_s'])/2,(b['speed_mps']-a['speed_mps'])/dt))
        else:accel.append((None,None))
    for a,b in zip(accel,accel[1:]):
        if a[0] is not None and b[0] is not None:jerk.append((b[1]-a[1])/(b[0]-a[0]))
    return [a[1] for a in accel if a[1] is not None],jerk

def evaluate_episode(ep):
    if ep.get('schema')!='driving-metrics-v1':raise ValueError('unsupported schema')
    rows=ep.get('samples',[]);acc,jerk=derivatives(rows);coverage=ep.get('coverage',{});events=ep.get('events',[])
    unique={};unkeyed=0
    for e in events:
        if not e.get('id'):unkeyed+=1;continue
        if e['id'] in unique and unique[e['id']]!=e:raise ValueError('conflicting duplicate event id')
        unique[e['id']]=e
    safety={}
    for kind in ['collision','offroad','red_light','intervention']:
        known=coverage.get(kind) is True and unkeyed==0
        safety[kind+'_events']=metric(sum(e.get('kind')==kind for e in unique.values()) if known else None,'count',None if known else 'missing instrumentation or event identity',len(unique))
    headway=[];ttc=[];eligible=0;nonclosing=0
    for r in rows:
        # Requires a validated in-path lead and bumper gap, not nearest object/center.
        if r.get('lead_in_path') is not True or r.get('lead_valid') is not True:continue
        gap=r.get('bumper_gap_m');speed=r.get('speed_mps');closing=r.get('closing_speed_mps')
        if not finite(gap) or gap<0:continue
        if finite(speed) and speed>.5:headway.append(gap/speed)
        if finite(closing):
            eligible+=1
            if closing>.1:ttc.append(gap/closing)
            else:nonclosing+=1
    safety['min_time_headway']=metric(min(headway) if headway else None,'s','requires valid in-path bumper gap and ego speed>0.5m/s' if not headway else None,len(headway))
    safety['min_ttc']=metric(min(ttc) if ttc else None,'s','no eligible closing lead' if not ttc else None,len(ttc));safety['ttc_nonclosing_samples']=metric(nonclosing if eligible else None,'count',None if eligible else 'no eligible lead measurements',n=eligible)
    dynamics={'mean_abs_speed_accel':metric(statistics.mean(map(abs,acc)) if acc else None,'m/s^2','insufficient contiguous speed/time samples' if not acc else None,len(acc)), 'mean_abs_speed_jerk':metric(statistics.mean(map(abs,jerk)) if jerk else None,'m/s^3','insufficient contiguous speed/time samples' if not jerk else None,len(jerk))}
    for key,unit in [('lateral_accel_mps2','m/s^2'),('curvature_per_m','1/m'),('tracking_error_m','m')]:
        vals=[abs(r[key]) for r in rows if finite(r.get(key))];dynamics['max_abs_'+key]=metric(max(vals) if vals else None,unit,'not observed' if not vals else None,len(vals))
    progress=[r.get('route_progress_m') for r in rows];known=bool(rows) and all(finite(x) for x in progress)
    latency={}
    for e in ep.get('planning_events',[]):
        if e.get('id') and finite(e.get('wall_ms')):
            if e['id'] in latency and latency[e['id']]!=e['wall_ms']:raise ValueError('conflicting planning id')
            latency[e['id']]=e['wall_ms']
    vals=list(latency.values())
    result={'schema':'driving-metrics-result-v1','scenario_fingerprint':ep.get('scenario_fingerprint'),'episode_status':ep.get('status','unknown'),'sample_count':len(rows),'safety':safety,'feasibility_comfort':dynamics,'efficiency':{'route_progress_delta':metric(progress[-1]-progress[0] if known else None,'m','requires consistent route progress coordinate' if not known else None,len(rows)),'route_completed':metric(ep.get('route_completed') if isinstance(ep.get('route_completed'),bool) else None,'bool','duration reached does not imply route completion' if ep.get('route_completed') is None else None)},'latency':{'planning_p95':metric(percentile(vals,.95) if vals else None,'ms','no uniquely identified planning samples' if not vals else None,len(vals))},'world_prediction':metric(None,'multi','not connected'),'cloud_llm':metric(None,'multi','not connected'),'overall_driving_pass':None,'limitations':['No aggregate driving score or acceptance without predefined scenario-specific safety/feasibility gates.','Scalar-speed derivatives are not full vector/lateral dynamics.','No eligible TTC is not a zero-risk certification.','Event counts are callbacks with stable identity, not deduced impact episodes.']}
    return result

def paired_comparison(a,b):
    if not a.get('scenario_fingerprint') or a['scenario_fingerprint']!=b.get('scenario_fingerprint'):raise ValueError('same complete scenario fingerprint required')
    return {'comparable_scenario':True,'overall_winner':None,'note':'Compare safety/efficiency/feasibility dimensions, no ADE-based winner'}


def scenario_fingerprint(config):
    required=['simulator_version','map','weather','traffic_seed','route','initial_actors','fixed_delta_seconds','sensor_config','controller_version']
    missing=[k for k in required if k not in config or config[k] is None]
    if missing:raise ValueError('Incomplete matched scenario: '+','.join(missing))
    return hashlib.sha256(json.dumps({k:config[k] for k in required},sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()
