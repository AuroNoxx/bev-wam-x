"""Read-only byte verification of obsolete OCC copies against retained archives."""
import hashlib,json,tarfile,time
from pathlib import Path


def digest(stream):
    h=hashlib.sha256()
    while chunk:=stream.read(1024*1024):h.update(chunk)
    return h.hexdigest()


def main():
    root=Path(__file__).resolve().parents[1]
    archives=root/'data/public/bench2drive/archives'
    clips=root/'data/public/bench2drive/clips'
    records=[];errors=[];archive_records=[]
    for archive in sorted(archives.glob('*.tar.gz')):
        route=archive.name.removesuffix('.tar.gz')
        if not (clips/route/'occ').exists():continue
        stat=archive.stat();count=0
        try:
            with tarfile.open(archive,'r|gz') as tar:
                for member in tar:
                    parts=Path(member.name).parts
                    if not member.isfile() or len(parts)!=3 or parts[0]!=route or parts[1]!='occ' or not parts[2].endswith('.npz'):continue
                    local=clips.joinpath(*parts)
                    if not local.is_file() or local.is_symlink():continue
                    ls=local.stat()
                    if ls.st_size!=member.size:raise ValueError(f'Size mismatch: {local}')
                    with tar.extractfile(member) as stream:a=digest(stream)
                    with local.open('rb') as stream:b=digest(stream)
                    if a!=b:raise ValueError(f'Hash mismatch: {local}')
                    records.append({'path':str(local),'bytes':ls.st_size,'mtime_ns':ls.st_mtime_ns,
                                    'sha256':a,'archive':str(archive),'member':member.name})
                    count+=1
            archive_records.append({'path':str(archive),'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns})
        except Exception as exc:errors.append({'archive':str(archive),'error':repr(exc)})
        print(json.dumps({'archive':archive.name,'verified_files':count,'total_GiB':sum(r['bytes'] for r in records)/2**30}),flush=True)
    report={'time':time.time(),'phase':'verified' if not errors else 'failed','purpose':'remove obsolete extracted OCC copies only; original archives, annotations, RGB, logs and results retained',
            'records':records,'archives':archive_records,'errors':errors,'bytes':sum(r['bytes'] for r in records)}
    (root/'artifacts/v5-qwen-drive-world/duplicate-occ-audit.json').write_text(json.dumps(report,indent=2))
    assert not errors,errors


if __name__=='__main__':main()
