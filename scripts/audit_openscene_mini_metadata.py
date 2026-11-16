import tarfile,pickle,io,numpy as np,json,importlib,hashlib,time
from pathlib import Path
class Safe(pickle.Unpickler):
    def find_class(self,module,name):
        allowed={('numpy','ndarray'),('numpy','dtype'),('numpy.core.multiarray','_reconstruct'),('numpy.core.multiarray','scalar'),('numpy._core.multiarray','_reconstruct'),('numpy._core.multiarray','scalar'),('numpy.core.numeric','_frombuffer'),('numpy._core.numeric','_frombuffer')}
        if (module,name) not in allowed:raise pickle.UnpicklingError('Blocked global '+module+'.'+name)
        return getattr(importlib.import_module(module),name)
root=Path(__file__).resolve().parents[1];p=root/'data/public/openscene-mini-audit/openscene_metadata_mini.tgz';started=time.perf_counter();logs=[];speeds=[];dts=[];deltas=[];occ=0;total=0;dups=set();seen=set()
with tarfile.open(p,'r|gz') as t:
    for m in t:
        if not m.isfile() or not m.name.endswith('.pkl'):continue
        assert m.size<256*1024**2
        data=Safe(io.BytesIO(t.extractfile(m).read())).load();data=sorted(data,key=lambda x:x['timestamp']);ts=np.array([x['timestamp'] for x in data],dtype=np.int64)/1e6;dt=np.diff(ts);v=np.array([np.linalg.norm(np.asarray(x['ego_dynamic_state'][:2],float))*3.6 for x in data]);pos=np.stack([x['ego2global_translation'][:2] for x in data]);fd=np.linalg.norm(np.diff(pos,axis=0),axis=1)/np.maximum(dt,1e-9)*3.6;valid=(dt>.4)&(dt<.6);deltas.extend((np.abs(fd-(v[:-1]+v[1:])/2)[valid]).tolist());dts.extend(dt.tolist());speeds.extend(v.tolist());total+=len(data)
        for x in data:
            if x['token'] in seen:dups.add(x['token'])
            seen.add(x['token']);occ+=bool(x.get('occ_gt_final_path'))
        high={}
        for threshold in [60,80,100,120]:
            spans=[];start=None
            for j in range(len(v)):
                if v[j]>=threshold and (start is None or .4<ts[j]-ts[j-1]<.6):
                    if start is None:start=j
                else:
                    if start is not None:spans.append(float(ts[j-1]-ts[start]));start=None
                    if v[j]>=threshold:start=j
            if start is not None:spans.append(float(ts[-1]-ts[start]))
            high[str(threshold)]={'frames':int((v>=threshold).sum()),'max_contiguous_seconds':max(spans,default=0),'segments_ge5s':sum(s>=5 for s in spans)}
        logs.append({'file':m.name,'frames':len(data),'duration_seconds':float(ts[-1]-ts[0]),'map':data[0].get('map_location'),'max_speed_kmh':float(v.max()),'gap_count_outside_0_4_to_0_6s':int((~valid).sum()),'highspeed':high});print('LOG',len(logs),len(data),round(float(v.max()),2),flush=True)
v=np.array(speeds);dt=np.array(dts);result={'source_url':'https://www.modelscope.cn/datasets/OpenDriveLab/OpenScene/resolve/master/openscene-v1.1/openscene_metadata_mini.tgz','archive_bytes':p.stat().st_size,'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest(),'logs':logs,'recordings':len(logs),'frames':total,'unique_tokens':len(seen),'duplicate_tokens':len(dups),'frames_with_occ_path_string':occ,'occ_files_downloaded':False,'velocity_interpretation':'norm first two ego_dynamic_state components, assumed m/s pending source definition; validated against pose finite differences','speed_kmh_percentiles':{str(q):float(np.percentile(v,q)) for q in [0,50,90,95,99,100]},'thresholds':{str(k):{'frames':int((v>=k).sum()),'recordings':sum(x['highspeed'][str(k)]['frames']>0 for x in logs),'max_contiguous_seconds':max(x['highspeed'][str(k)]['max_contiguous_seconds'] for x in logs),'segments_ge5s':sum(x['highspeed'][str(k)]['segments_ge5s'] for x in logs)} for k in [60,80,100,120]},'timestamp_delta_seconds_percentiles':{str(q):float(np.percentile(dt,q)) for q in [0,50,95,100]},'velocity_vs_pose_fd_abs_diff_kmh_median_p95':[float(np.percentile(deltas,q)) for q in [50,95]],'audit_seconds':time.perf_counter()-started,'limits':'Mini metadata only; speed is not road class; no OCC visibility/source causality certification; no project test files read; no pickle custom globals permitted except numpy reconstruction.'}
out=root/'artifacts/openscene-mini-metadata-audit-20261115.json';assert not out.exists();out.write_text(json.dumps(result,indent=2));print('SUMMARY',json.dumps({k:v for k,v in result.items() if k!='logs'}),flush=True)
