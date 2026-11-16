"""Resume project-local CARLA chunks; download only, no installation."""
from pathlib import Path
import argparse, concurrent.futures, json, subprocess, time, threading
ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/'.cache/carla-0915'
URL='https://downloads.carlasim.com/Windows/CARLA_0.9.15.zip'
SIZE=7755135185
CHUNK=256*1024*1024
STOP=threading.Event()
LOCK=threading.Lock()
CHILDREN=set()

def fetch(job):
    path,start,end=job
    needed=end-start+1
    if path.exists() and path.stat().st_size==needed: return needed
    header=path.with_suffix('.headers')
    for attempt in range(4):
        if STOP.is_set(): raise RuntimeError('Download stopped')
        cmd=['curl.exe','--fail','--silent','--show-error','--location',
             '--connect-timeout','20','--max-time','300','--speed-limit','131072','--speed-time','30',
             '--header','If-Match: "9f7a0c3973fc1a5fd6e327cd100e7dc4-116"',
             '--dump-header',str(header),'--range',f'{start}-{end}','--output',str(path),URL]
        with LOCK:
            if STOP.is_set(): raise RuntimeError('Download stopped')
            proc=subprocess.Popen(cmd); CHILDREN.add(proc)
        code=proc.wait()
        with LOCK: CHILDREN.discard(proc)
        headers=header.read_text(errors='replace').lower() if header.exists() else ''
        expected=f'content-range: bytes {start}-{end}/{SIZE}'
        if code==0 and path.exists() and path.stat().st_size==needed and expected in headers: return needed
        print(f'Retrying range {start}-{end}, attempt {attempt+1}/4',flush=True)
        if STOP.wait(2): raise RuntimeError('Download stopped')
    raise RuntimeError(f'Failed range {start}-{end}; completed chunks preserved')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--connections',type=int,choices=[4,8,16],default=16)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    CACHE.mkdir(parents=True,exist_ok=True)
    dest=CACHE/'CARLA_0.9.15.zip'
    if dest.exists() and dest.stat().st_size==SIZE:
        print('Full-sized archive exists; wrapper will verify integrity.'); return
    parts=[(i,start,min(start+CHUNK,SIZE)-1) for i,start in enumerate(range(0,SIZE,CHUNK))]
    jobs=[]; sources=[]; done=0
    for index,start,end in parts:
        path=CACHE/f'part-{index:04d}.bin'
        have=path.stat().st_size if path.exists() else 0
        if have>end-start+1: raise RuntimeError(f'Oversized existing segment: {path}')
        if have: sources.append((path,have))
        done+=have
        for offset in range(start+have,end+1,32*1024*1024):
            next_end=min(offset+32*1024*1024-1,end)
            segment=CACHE/f'remainder-{offset}-{next_end}.bin'
            size=next_end-offset+1
            sources.append((segment,size))
            if segment.exists() and segment.stat().st_size==size: done+=size
            else: jobs.append((segment,offset,next_end))
    print(f'Reusing {done/1e9:.2f} / {SIZE/1e9:.2f} GB; remaining chunks: {len(jobs)}',flush=True)
    if args.dry_run: return
    started=time.monotonic(); initial=done
    pool=concurrent.futures.ThreadPoolExecutor(max_workers=args.connections)
    try:
        futures=[pool.submit(fetch,job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            done+=future.result(); elapsed=time.monotonic()-started
            rate=(done-initial)/max(elapsed,.01)
            status={'downloaded_bytes':done,'total_bytes':SIZE,'elapsed_s':elapsed,'bytes_per_second':rate}
            (ROOT/'logs/bench2drive-0915/download-status.json').write_text(json.dumps(status))
            print(f'{done/SIZE:6.1%}  {done/1e9:.2f}/{SIZE/1e9:.2f} GB  {rate/1e6:.1f} MB/s',flush=True)
    except BaseException:
        STOP.set()
        with LOCK:
            for proc in list(CHILDREN):
                if proc.poll() is None: proc.terminate()
        pool.shutdown(wait=True,cancel_futures=True)
        raise
    else: pool.shutdown(wait=True)
    print('Combining verified-size chunks...',flush=True)
    staged=CACHE/'CARLA_0.9.15.zip.assembling'
    with staged.open('wb') as out:
        for path,expected in sources:
            if path.stat().st_size!=expected: raise RuntimeError(f'Source size changed: {path}')
            with path.open('rb') as stream:
                while data:=stream.read(8*1024*1024): out.write(data)
    if staged.stat().st_size!=SIZE: raise RuntimeError('Final size mismatch')
    staged.replace(dest)
    print('DOWNLOAD COMPLETE:',dest,flush=True)

if __name__=='__main__': main()

