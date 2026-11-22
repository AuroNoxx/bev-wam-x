"""Resumable binary streaming sync over `colab ssh` for corpus shards, bundles, and weights."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


DEFAULT_COLAB_PY = r'D:/ai/envs/uv-tools/google-colab-cli/Scripts/python.exe'
DEFAULT_KEY = str(Path.home() / '.ssh' / 'id_ed25519_colab')


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _ssh_cmd(colab: str, session: str, key: str, remote_cmd: str):
    key_fwd = Path(key).as_posix()
    colab_py = Path(colab).as_posix()
    proxy = f'{colab_py} -m colab_cli.cli ssh --proxy-mode -s {session} --identity {key_fwd} --colab-internal-mode cmd'
    return [
        'ssh', '-T',
        '-o', f'ProxyCommand={proxy}',
        '-o', 'StrictHostKeyChecking=no',
        '-o', f'UserKnownHostsFile={os.devnull if hasattr(os, "devnull") else "NUL"}',
        '-o', 'BatchMode=yes',
        '-o', 'LogLevel=ERROR',
        '-i', key_fwd,
        'root@colab-runtime',
        f'cd /content 2>/dev/null; {remote_cmd}',
    ]


def run_ssh(colab: str, session: str, key: str, remote_cmd: str, stdin_file=None, check=True):
    cmd = _ssh_cmd(colab, session, key, remote_cmd)
    res = subprocess.run(cmd, stdin=stdin_file if stdin_file is not None else subprocess.DEVNULL, capture_output=True)
    if check and res.returncode != 0:
        err = res.stderr.decode('utf-8', errors='replace')
        out = res.stdout.decode('utf-8', errors='replace')
        raise RuntimeError(f'SSH command failed ({res.returncode}): {remote_cmd}\nSTDOUT: {out}\nSTDERR: {err}')
    return res


def upload_file(colab: str, session: str, key: str, local_path: Path, remote_path: str):
    if not local_path.is_file():
        raise FileNotFoundError(local_path)
    digest = sha256_file(local_path)
    size = local_path.stat().st_size
    rdir = str(Path(remote_path).parent).replace('\\', '/')
    if not rdir.startswith('/'):
        rdir = '/' + rdir.lstrip('/')
    q_dir = shlex.quote(rdir)
    q_tmp = shlex.quote(f'{remote_path}.tmp')
    q_dst = shlex.quote(remote_path)
    t0 = time.perf_counter()
    with local_path.open('rb') as f:
        run_ssh(colab, session, key, f'mkdir -p {q_dir} && cat > {q_tmp} && mv {q_tmp} {q_dst}', stdin_file=f)
    res = run_ssh(colab, session, key, f'sha256sum {q_dst}')
    remote_sha = res.stdout.decode('utf-8').strip().split()[0]
    if remote_sha != digest:
        raise RuntimeError(f'SHA-256 mismatch for {remote_path}: local={digest} remote={remote_sha}')
    dt = max(time.perf_counter() - t0, 1e-3)
    print(f'[sync] uploaded {local_path.name} -> {remote_path} ({size / (1 << 20):.2f} MiB, {size / 1e6 / dt:.2f} MB/s, sha256={digest[:12]})', flush=True)
    return size / 1e6 / dt


def download_file(colab: str, session: str, key: str, remote_path: str, local_path: Path):
    q_src = shlex.quote(remote_path)
    sha_res = run_ssh(colab, session, key, f'sha256sum {q_src}')
    remote_sha = sha_res.stdout.decode('utf-8').strip().split()[0]
    local_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = local_path.with_name(f'{local_path.name}.tmp')
    cmd = _ssh_cmd(colab, session, key, f'cat {q_src}')
    t0 = time.perf_counter()
    with tmp_path.open('wb') as out_f:
        res = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=out_f, stderr=subprocess.PIPE)
    if res.returncode != 0:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(f'Download failed ({res.returncode}): {res.stderr.decode("utf-8", errors="replace")}')
    local_sha = sha256_file(tmp_path)
    if local_sha != remote_sha:
        tmp_path.unlink()
        raise RuntimeError(f'Download SHA-256 mismatch for {remote_path}: remote={remote_sha} local={local_sha}')
    tmp_path.replace(local_path)
    size = local_path.stat().st_size
    dt = max(time.perf_counter() - t0, 1e-3)
    print(f'[sync] downloaded {remote_path} -> {local_path} ({size / (1 << 20):.2f} MiB, {size / 1e6 / dt:.2f} MB/s, sha256={local_sha[:12]})', flush=True)


def _upload_one_shard(colab: str, session: str, key: str, tar_path: Path, ep_id: str, expected_bytes: int, remote_corpus: str, retries: int = 3):
    q_corp = shlex.quote(remote_corpus)
    q_stg = shlex.quote(f'{remote_corpus}/.staging/{ep_id}')
    q_dst = shlex.quote(f'{remote_corpus}/episodes/{ep_id}')
    cmd_str = (
        f'rm -rf {q_stg} && mkdir -p {q_stg} {q_corp}/episodes && '
        f'tar xf - -C {q_stg} && '
        f'rm -rf {q_dst} && mv {q_stg}/episodes/{shlex.quote(ep_id)} {q_dst} && '
        f'rm -rf {q_stg} && test -f {q_dst}/episode.json'
    )
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            t0 = time.perf_counter()
            with tar_path.open('rb') as f:
                run_ssh(colab, session, key, cmd_str, stdin_file=f)
            dt = max(time.perf_counter() - t0, 1e-3)
            mb_s = expected_bytes / 1e6 / dt
            print(f'[sync] shard {tar_path.name} OK ({expected_bytes / (1 << 20):.1f} MiB in {dt:.1f}s = {mb_s:.2f} MB/s)', flush=True)
            return mb_s
        except Exception as err:
            last_err = err
            print(f'[sync] shard {tar_path.name} attempt {attempt}/{retries} failed: {err}', file=sys.stderr, flush=True)
            time.sleep(2.0 * attempt)
    raise RuntimeError(f'Shard {tar_path.name} failed after {retries} attempts: {last_err}')


def upload_corpus(colab: str, session: str, key: str, pack_dir: Path, remote_corpus: str,
                  only: list[str] | None = None, workers: int = 1):
    summary = json.loads((pack_dir / 'pack-sha256.json').read_text(encoding='utf-8'))
    files = summary['files']
    q_corp = shlex.quote(remote_corpus)
    existing_res = run_ssh(
        colab, session, key,
        f'mkdir -p {q_corp}/episodes && python3 -c "import pathlib; r=pathlib.Path({remote_corpus!r})/\'episodes\'; print(\'\\n\'.join(p.name for p in sorted(r.iterdir()) if (p/\'episode.json\').is_file()))"',
    )
    existing_eps = {line.strip() for line in existing_res.stdout.decode('utf-8').splitlines() if line.strip()}

    todo = []
    for tar_name in sorted(files):
        if only and tar_name not in only and files[tar_name]['episode'] not in only:
            continue
        rec = files[tar_name]
        if rec['episode'] in existing_eps:
            print(f'[sync] skip existing {tar_name}', flush=True)
            continue
        tar_path = pack_dir / tar_name
        if not tar_path.is_file():
            raise FileNotFoundError(tar_path)
        todo.append((tar_path, rec['episode'], rec['bytes']))

    if not todo:
        print('[sync] all requested shards already present remotely', flush=True)
        return []

    rates = []
    t_all = time.perf_counter()
    if workers <= 1 or len(todo) == 1:
        for tar_path, ep_id, sz in todo:
            rates.append(_upload_one_shard(colab, session, key, tar_path, ep_id, sz, remote_corpus))
    else:
        with ThreadPoolExecutor(max_workers=int(workers)) as pool:
            futs = [pool.submit(_upload_one_shard, colab, session, key, p, e, s, remote_corpus) for p, e, s in todo]
            for fut in as_completed(futs):
                rates.append(fut.result())
    tot_bytes = sum(s for _, _, s in todo)
    tot_dt = max(time.perf_counter() - t_all, 1e-3)
    print(f'[sync] completed {len(todo)} shards ({tot_bytes / (1 << 30):.2f} GiB) in {tot_dt:.1f}s (aggregate {tot_bytes / 1e6 / tot_dt:.2f} MB/s)', flush=True)
    return rates


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--colab', default=DEFAULT_COLAB_PY)
    p.add_argument('--session', '-s', default='stage26')
    p.add_argument('--key', '-i', default=DEFAULT_KEY)
    sub = p.add_subparsers(dest='cmd', required=True)

    up = sub.add_parser('upload-file')
    up.add_argument('local', type=Path)
    up.add_argument('remote')

    dl = sub.add_parser('download-file')
    dl.add_argument('remote')
    dl.add_argument('local', type=Path)

    uc = sub.add_parser('upload-corpus')
    uc.add_argument('pack_dir', type=Path)
    uc.add_argument('--remote-corpus', default='/content/corpus')
    uc.add_argument('--only', nargs='*', default=None)
    uc.add_argument('--workers', type=int, default=1)

    args = p.parse_args()
    if args.cmd == 'upload-file':
        upload_file(args.colab, args.session, args.key, args.local, args.remote)
    elif args.cmd == 'download-file':
        download_file(args.colab, args.session, args.key, args.remote, args.local)
    elif args.cmd == 'upload-corpus':
        upload_corpus(args.colab, args.session, args.key, args.pack_dir, args.remote_corpus, args.only, args.workers)
