"""Download a small, revision-pinned public-data subset; no self-collected data."""
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor
import json, hashlib, subprocess, time

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'data/public'
DEST.mkdir(parents=True,exist_ok=True)


def api(path):
    last=None
    for host in ['https://hf-mirror.com','https://huggingface.co']:
        try:
            with urlopen(Request(host+path,headers={'User-Agent':'highway-occ-flow/benchmark'}),timeout=45) as response:
                return json.load(response)
        except Exception as e: last=e
    raise last


def download(task):
    repo,rev,name,relative=task
    destination=DEST/relative
    destination.parent.mkdir(parents=True,exist_ok=True)
    if not destination.exists():
        partial=destination.with_suffix(destination.suffix+'.part')
        for host in ['https://hf-mirror.com','https://huggingface.co']:
            url=f'{host}/datasets/{repo}/resolve/{rev}/{quote(name)}'
            result=subprocess.run(['curl.exe','-L','--fail','--retry','4','--retry-all-errors','--retry-delay','1','--connect-timeout','20','--max-time','600','-sS','-C','-','-o',str(partial),url],capture_output=True,text=True)
            if result.returncode==0:
                partial.replace(destination)
                break
        else: raise RuntimeError(f'Download failed {name}: {result.stderr[-500:]}')
    digest=hashlib.sha256(destination.read_bytes()).hexdigest()
    return {'repo':repo,'revision':rev,'file':name,'local':str(destination.relative_to(ROOT)),
            'bytes':destination.stat().st_size,'sha256':digest}


def main():
    info=api('/api/datasets/tasl-lab/uniocc')
    rev=info['sha']
    names=[s['rfilename'] for s in info['siblings']]
    npzs=[n for n in names if n.startswith('Carla-10Hz-mini/') and n.endswith('.npz')]
    scenes=sorted(set(n.split('/')[1] for n in npzs))
    print('UniOcc revision',rev,'scenes',scenes,flush=True)
    tasks=[]
    for scene in scenes[:2]:
        frames=sorted([n for n in npzs if n.split('/')[1]==scene],key=lambda n:int(Path(n).stem))[:160]
        selected_ids={int(Path(n).stem) for n in frames}
        for name in frames:
            tasks.append(('tasl-lab/uniocc',rev,name,'uniocc/'+name))
        for name in names:
            if name.startswith('Carla-10Hz-mini/'+scene+'/') and '_camera_' in name and int(Path(name).name.split('_')[0]) in selected_ids:
                tasks.append(('tasl-lab/uniocc',rev,name,'uniocc/'+name))
    info_b=api('/api/datasets/rethinklab/Bench2Drive-V0.0.4')
    bnames=[s['rfilename'] for s in info_b['siblings']]
    (DEST/'bench2drive-file-list.json').write_text(json.dumps(info_b,indent=2),encoding='utf-8')
    # Two independent highway cut-in routes: inspect/train/validation throughput only.
    chosen=[n for n in bnames if n.startswith('HighwayCutIn_') and n.endswith('.tar.gz')][:2]
    for name in chosen:
        tasks.insert(0,('rethinklab/Bench2Drive-V0.0.4',info_b['sha'],name,'bench2drive/archives/'+name))
    (DEST/'download-plan.json').write_text(json.dumps(tasks,indent=2),encoding='utf-8')
    print('Downloading files',len(tasks),'B2D clips',chosen,flush=True)
    records=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for record in pool.map(download,tasks):
            records.append(record)
            if len(records)%20==0 or record['file'].endswith('.tar.gz'):
                print('Downloaded',len(records),'/',len(tasks),'bytes',sum(r['bytes'] for r in records),flush=True)
                (DEST/'download-manifest.partial.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    (DEST/'download-manifest.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    print('DONE',len(records),'files',sum(r['bytes'] for r in records),'bytes',flush=True)


if __name__=='__main__': main()
