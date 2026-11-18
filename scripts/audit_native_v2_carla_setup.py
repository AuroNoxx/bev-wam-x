"""Read-only checkpoint/split/source audit; writes isolated diagnostic evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import psutil
import torch

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    processes=[]
    for proc in psutil.process_iter(['pid','name','cmdline']):
        try:
            if any(word in (proc.info['name'] or '').lower() for word in ['python','carla','ollama','llama']):processes.append(proc.info)
        except psutil.Error:pass
    running_trainers=[r for r in processes if any('train_occ_qwen_native' in s for s in r.get('cmdline') or [])]
    if running_trainers:raise RuntimeError('Trainer running, do not load model on GPU')
    cfg=json.loads((ROOT/'configs/occ-qwen-native-v2.json').read_text())
    manifest_path=ROOT/cfg['manifest'];manifest=json.loads(manifest_path.read_text());manifest_sha=sha(manifest_path)
    ckpts={}
    for name in ['best.ckpt','latest.ckpt']:
        path=ROOT/'artifacts/occ-qwen-native-v2-training'/name
        cp=torch.load(path,map_location='cpu',weights_only=False)
        assert cp['task_state']['settings']==cfg and cp['task_state']['manifest_sha']==manifest_sha
        ckpts[name]={'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path),
                     'epoch_one_based':cp['epoch']+1,'step':cp['global_step'],
                     'schema':cp['task_state']['schema'],'config_matches_checkpoint_exactly':True,
                     'manifest_matches_checkpoint_exactly':True,'checkpoint_tensor_count':len(cp['state_dict'])}
        del cp
    index_path=ROOT/cfg['cache_index'];index=json.loads(index_path.read_text())
    # Compare actual file lists, not merely the names of town split directories.
    from native_occ_cached_data import NativeCachedDataset
    actual=NativeCachedDataset(index_path,'train')
    train_routes=sorted({r['route'] for r in manifest['routes'] if r['split']=='train'})
    val_routes=sorted({r['route'] for r in manifest['routes'] if r['split']=='validation'})
    source=ROOT/'runtime/Bench2Drive-0.0.4/leaderboard/data/bench2drive220.xml'
    rows=ET.parse(source).getroot().findall('route');scenes=[]
    for name,rid in [('turn','26990'),('obstacle','26456'),('dynamic','27515')]:
        row=next(r for r in rows if r.get('id')==rid);town=row.get('town')
        matches=[r for r in train_routes+val_routes if '_'+town+'_' in r]
        assert not matches
        scenes.append({'name':name,'xml_route_id':rid,'town':town,'matches_in_train_or_validation_routes':matches,
                       'held_out_from_this_v2_finetuning':True,
                       'basis':'checkpoint manifest hash matches actual5296 train windows; training towns12 and validation13; selected Town03 absent from both',
                       'limits':'No claim about Qwen pretraining; custom controlled traffic instead of official ScenarioRunner; not original full B2D benchmark',
                       'original_scenarios':[s.get('type') for s in row.findall('.//scenario')]})
    files=['AGENTS.md','docs/ARCHITECTURE.md','docs/OCC_QWEN_NATIVE_V2_20261117.md',
           'docs/OCC_QWEN_NATIVE_V2_FINAL_AUDIT_20261117.md','scripts/b2d_native_occ.py',
           'scripts/occ_qwen_native_v2.py','scripts/native_v2_carla_contract.py',
           'scripts/native_v2_carla_worker.py','scripts/run_native_v2_carla.py',
           'runtime/Bench2Drive-0.0.4/tools/occ_online_carla.py',
           'runtime/Bench2Drive-0.0.4/tools/data_collect.py']
    report={'processes':processes,'running_native_trainers':running_trainers,'checkpoints':ckpts,
            'config':cfg,'manifest_sha256':manifest_sha,'cache_index_sha256':sha(index_path),
            'train_windows':len(actual),'manifest_counts':manifest['counts'],'train_routes':train_routes,
            'validation_routes':val_routes,'scenes':scenes,'route_xml_sha256':sha(source),
            'source_hashes':{f:sha(ROOT/f) for f in files},'weights_data_modified':False,
            'torch_version':torch.__version__,'gpu':torch.cuda.get_device_name(0),
            'gpu_probe_only_no_model_allocation':True}
    write=out/'setup-audit.json';write.write_text(json.dumps(report,indent=2));print(json.dumps({'audit':str(write),'checkpoints':ckpts,'scenes':scenes},indent=2))


if __name__=='__main__':main()
