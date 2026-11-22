"""Zero-CU Google Drive upload and verification helper for Stage 2.6 Colab packs."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
from typing import Any


ROOT_FILES = (
    'code_bundle.tar.gz',
    'manifest.jsonl',
    'pack-sha256.json',
)
FOLDER_MIME = 'application/vnd.google-apps.folder'


def _gdrive(args: list[str], check: bool = True) -> str:
    """Invoke the bare `gdrive` plugin CLI (carries OAuth session credentials)."""
    exe = shutil.which('gdrive')
    if not exe:
        raise RuntimeError('`gdrive` executable not found on PATH')
    proc = subprocess.run(
        [exe, *args],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
    )
    if check and proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or '').strip()
        raise RuntimeError(f'gdrive {" ".join(args)} failed (rc={proc.returncode}): {msg}')
    return proc.stdout


_ID_RE = re.compile(r'\(ID:\s*([A-Za-z0-9_-]+)\)')


def _parse_json(raw: str, default: Any) -> Any:
    """Parse JSON from `gdrive --json` stdout, or extract `(ID: ...)` / fallback."""
    text = (raw or '').strip()
    if not text or text == 'null':
        return default
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        for line in reversed(text.splitlines()):
            s = line.strip()
            if s.startswith('{') or s.startswith('['):
                try:
                    return json.loads(s)
                except json.JSONDecodeError:
                    pass
        if isinstance(default, dict):
            m = _ID_RE.search(text)
            if m:
                return {'id': m.group(1)}
        return default


def list_folder(folder_id: str | None = None) -> list[dict[str, Any]]:
    """List non-trashed children inside `folder_id` (or Drive root when None)."""
    args = ['readonly', 'ls']
    if folder_id:
        args.append(folder_id)
    args.extend(['--max', '0', '--json'])
    data = _parse_json(_gdrive(args), [])
    if data is None:
        return []
    if not isinstance(data, list):
        raise RuntimeError(f'unexpected ls --json payload: {type(data).__name__}')
    return data


def _index_by_name(items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_name: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        name = str(item.get('name', ''))
        if name:
            by_name.setdefault(name, []).append(item)
    return by_name


def ensure_folder(name: str, parent_id: str | None = None) -> str:
    """Return existing Drive folder ID or create it under `parent_id`."""
    for item in list_folder(parent_id):
        if item.get('name') == name and item.get('mimeType') == FOLDER_MIME:
            return str(item['id'])
    args = ['mutate', 'mkdir', name]
    if parent_id:
        args.extend(['--parent', parent_id])
    args.append('--json')
    created = _parse_json(_gdrive(args), {})
    if isinstance(created, dict) and created.get('id'):
        return str(created['id'])
    for item in list_folder(parent_id):
        if item.get('name') == name and item.get('mimeType') == FOLDER_MIME:
            return str(item['id'])
    raise RuntimeError(f'failed to resolve created folder {name!r}')


def _remote_size(file_id: str | None, name: str, parent_id: str, payload: Any) -> int:
    """Resolve remote byte size from upload JSON, falling back to `info` or `ls`."""
    if isinstance(payload, dict):
        if payload.get('size') is not None:
            return int(payload['size'])
        if not file_id and payload.get('id'):
            file_id = str(payload['id'])
    if file_id:
        info = _parse_json(_gdrive(['readonly', 'info', file_id, '--json']), {})
        if isinstance(info, dict) and info.get('size') is not None:
            return int(info['size'])
    matches = [it for it in list_folder(parent_id) if it.get('name') == name and it.get('mimeType') != FOLDER_MIME]
    if len(matches) == 1 and matches[0].get('size') is not None:
        return int(matches[0]['size'])
    return -1


def upload_one(
    local_path: Path,
    parent_id: str,
    existing_by_name: dict[str, dict[str, Any]],
    lock: threading.Lock,
    retries: int = 3,
) -> tuple[str, int, float, bool]:
    """Upload or update one file with exact-size skip and retry without duplicates."""
    name = local_path.name
    local_size = int(local_path.stat().st_size)
    mib = local_size / (1024 * 1024)

    with lock:
        existing = existing_by_name.get(name)
    if existing is not None and existing.get('size') is not None and int(existing['size']) == local_size:
        print(f'[drive] skip {name} ({mib:.2f} MiB, size match)', flush=True)
        return name, local_size, 0.0, True

    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        with lock:
            existing = existing_by_name.get(name)
        t0 = time.monotonic()
        try:
            if existing is not None and existing.get('id'):
                file_id = str(existing['id'])
                raw = _gdrive(['mutate', 'update', file_id, str(local_path), '--json'])
            else:
                file_id = None
                raw = _gdrive(['mutate', 'upload', str(local_path), '--parent', parent_id, '--json'])
            payload = _parse_json(raw, {})
            if isinstance(payload, dict) and payload.get('id'):
                file_id = str(payload['id'])
            rsize = _remote_size(file_id, name, parent_id, payload)
            if rsize != local_size:
                raise RuntimeError(f'remote size mismatch for {name}: got {rsize}, expected {local_size}')
            dt = max(time.monotonic() - t0, 1e-6)
            mbps = (local_size / 1e6) / dt
            with lock:
                existing_by_name[name] = {
                    'id': file_id,
                    'name': name,
                    'size': str(rsize),
                }
            print(f'[drive] OK {name} ({mib:.2f} MiB in {dt:.1f}s = {mbps:.2f} MB/s)', flush=True)
            return name, local_size, dt, False
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            with lock:
                matches = [
                    it for it in list_folder(parent_id)
                    if it.get('name') == name and it.get('mimeType') != FOLDER_MIME
                ]
                if len(matches) == 1:
                    existing_by_name[name] = matches[0]
                elif len(matches) > 1:
                    raise RuntimeError(f'duplicate remote entries for {name} in {parent_id}: {len(matches)}') from exc
            if attempt < retries:
                time.sleep(5 * attempt)
    raise RuntimeError(f'failed uploading {name} after {retries} attempts: {last_err}') from last_err


def _expected_files(pack_dir: Path, stage2_weights: Path, only: list[str] | None = None) -> tuple[list[Path], list[Path]]:
    control = [pack_dir / name for name in ROOT_FILES] + [stage2_weights]
    for p in control:
        if not p.is_file():
            raise FileNotFoundError(f'missing required upload file: {p}')
    shards = sorted(pack_dir.glob('*.tar'))
    if only:
        wanted = set(only)
        shards = [p for p in shards if p.name in wanted]
        missing = sorted(wanted - {p.name for p in shards})
        if missing:
            raise FileNotFoundError(f'--only shard(s) not found in {pack_dir}: {missing}')
    return control, shards


def _verify_items(targets: list[Path], folder_id: str, label: str) -> int:
    by_name = _index_by_name([it for it in list_folder(folder_id) if it.get('mimeType') != FOLDER_MIME])
    errors: list[str] = []
    ok = 0
    for path in targets:
        matches = by_name.get(path.name, [])
        expected = int(path.stat().st_size)
        if len(matches) == 0:
            errors.append(f'missing {path.name}')
        elif len(matches) > 1:
            errors.append(f'duplicate {path.name} ({len(matches)} copies)')
        else:
            got = int(matches[0].get('size', -1))
            if got != expected:
                errors.append(f'size mismatch {path.name}: remote={got} local={expected}')
            else:
                ok += 1
    if errors:
        raise RuntimeError(f'{label} verification failed: {"; ".join(errors)}')
    return ok


def cmd_upload_pack(args: argparse.Namespace) -> int:
    pack_dir = Path(args.pack_dir)
    stage2_weights = Path(args.stage2_weights)
    control_files, shard_files = _expected_files(pack_dir, stage2_weights, args.only)

    root_id = ensure_folder(args.root_name)
    shards_id = ensure_folder('shards', parent_id=root_id)
    runs_id = ensure_folder('runs', parent_id=root_id)
    stage26_run_id = ensure_folder('stage26', parent_id=runs_id)
    print(
        f'[drive] folders: root={root_id} shards={shards_id} runs/stage26={stage26_run_id}',
        flush=True,
    )

    root_by_name = {
        k: v[0]
        for k, v in _index_by_name([it for it in list_folder(root_id) if it.get('mimeType') != FOLDER_MIME]).items()
        if len(v) == 1
    }
    root_lock = threading.Lock()
    for p in control_files:
        upload_one(p, root_id, root_by_name, root_lock)
    root_ok = _verify_items(control_files, root_id, 'control')
    print(f'VERIFIED_CONTROL: {root_ok}/{len(control_files)}', flush=True)

    if args.control_only:
        return 0

    shards_existing_all = _index_by_name([it for it in list_folder(shards_id) if it.get('mimeType') != FOLDER_MIME])
    dupes = [k for k, v in shards_existing_all.items() if len(v) > 1]
    if dupes:
        raise RuntimeError(f'duplicate files already present in shards/: {dupes}')
    shards_by_name = {k: v[0] for k, v in shards_existing_all.items()}
    shards_lock = threading.Lock()

    t_all = time.monotonic()
    uploaded_bytes = 0
    workers = max(1, int(args.workers))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(upload_one, p, shards_id, shards_by_name, shards_lock) for p in shard_files]
        for fut in as_completed(futs):
            _name, size, _dt, skipped = fut.result()
            if not skipped:
                uploaded_bytes += size

    ok = _verify_items(shard_files, shards_id, 'shards')
    dt_all = max(time.monotonic() - t_all, 1e-6)
    agg_mbps = (uploaded_bytes / 1e6) / dt_all
    print(
        f'VERIFIED_SHARDS: {ok}/{len(shard_files)} (uploaded {uploaded_bytes / (1024 ** 3):.2f} GiB in {dt_all:.1f}s = {agg_mbps:.2f} MB/s)',
        flush=True,
    )
    return 0


def cmd_verify_pack(args: argparse.Namespace) -> int:
    pack_dir = Path(args.pack_dir)
    stage2_weights = Path(args.stage2_weights)
    control_files, shard_files = _expected_files(pack_dir, stage2_weights, None)

    root_id = ensure_folder(args.root_name)
    shards_id = ensure_folder('shards', parent_id=root_id)
    root_ok = _verify_items(control_files, root_id, 'control')
    shard_ok = _verify_items(shard_files, shards_id, 'shards')
    total_bytes = sum(p.stat().st_size for p in [*control_files, *shard_files])
    print(
        f'VERIFY_PACK_OK: control={root_ok}/{len(control_files)} shards={shard_ok}/{len(shard_files)} '
        f'total_gib={total_bytes / (1024 ** 3):.2f}',
        flush=True,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Zero-CU Google Drive pack upload/verification')
    sub = p.add_subparsers(dest='cmd', required=True)

    for name, has_upload_args in (('upload-pack', True), ('verify-pack', False)):
        sp = sub.add_parser(name)
        sp.add_argument('--pack-dir', default='data/v2-mapless-wa/colab-pack-v26')
        sp.add_argument('--stage2-weights', default='models/v2_stage2_nativeC_40k_20261121/stage2.pt')
        sp.add_argument('--root-name', default='highway_occ_flow_v26')
        if has_upload_args:
            sp.add_argument('--only', nargs='*', default=None)
            sp.add_argument('--workers', type=int, default=2)
            sp.add_argument('--control-only', action='store_true')
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == 'upload-pack':
        return cmd_upload_pack(args)
    if args.cmd == 'verify-pack':
        return cmd_verify_pack(args)
    raise ValueError(args.cmd)


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
