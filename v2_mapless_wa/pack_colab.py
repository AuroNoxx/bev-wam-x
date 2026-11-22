"""Pack released CARLA corpora into per-episode uncompressed tar shards (PNG or bit-exact lossless WebP)."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import tarfile

import cv2
import numpy as np


RGB_NAMES = ('front.png', 'front_left.png', 'front_right.png')


def sanitize_tag(name: str) -> str:
    return ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in name)


def parse_root_arg(spec: str):
    if '=' in spec:
        tag, raw = spec.split('=', 1)
        return sanitize_tag(tag), Path(raw)
    p = Path(spec)
    return sanitize_tag(p.name), p


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _encode_frame_webp(frame_dir: Path, verify_pixel_exact: bool = False):
    out = []
    for png_name in RGB_NAMES:
        src = frame_dir / png_name
        bgr = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f'Failed to read RGB frame: {src}')
        ok, buf = cv2.imencode('.webp', bgr, [cv2.IMWRITE_WEBP_QUALITY, 101])
        if not ok:
            raise RuntimeError(f'Lossless WebP encoding failed for {src}')
        raw = buf.tobytes()
        if verify_pixel_exact:
            dec = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if dec is None or not np.array_equal(bgr, dec):
                raise RuntimeError(f'Lossless WebP verification failed for {src}')
        webp_rel = f"{frame_dir.name}/{png_name[:-4]}.webp"
        out.append((webp_rel, raw, src.stat().st_mtime))
    return out


def pack(roots, out_dir: Path, limit_episodes: int | None = None,
         image_codec: str = 'webp_lossless', workers: int = 8, resume: bool = False):
    out_dir.mkdir(parents=True, exist_ok=True)
    seen_tags = set()
    tagged_roots = []
    for item in roots:
        tag, path = parse_root_arg(item)
        if not path.is_dir():
            raise FileNotFoundError(f'Missing corpus root: {path}')
        if tag in seen_tags:
            raise ValueError(f'Duplicate corpus tag {tag!r}; specify tag=path explicitly')
        seen_tags.add(tag)
        tagged_roots.append((tag, path))

    existing_summary = {}
    summary_path = out_dir / 'pack-sha256.json'
    if resume and summary_path.is_file():
        existing_data = json.loads(summary_path.read_text(encoding='utf-8'))
        if existing_data.get('image_codec', 'png') == image_codec:
            existing_summary = existing_data.get('files', {})

    manifest_lines = []
    shard_records = {}
    packed_eps = 0
    total_bytes = 0
    total_frames = 0
    verified_lossless = False

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        for tag, root in tagged_roots:
            manifest_path = root / 'manifest.jsonl'
            if not manifest_path.is_file():
                raise FileNotFoundError(f'Missing manifest.jsonl in {root}')
            rows_by_ep = {}
            with manifest_path.open('r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    rows_by_ep.setdefault(row['episode'], []).append(row)

            for ep_id in sorted(rows_by_ep):
                if limit_episodes is not None and packed_eps >= limit_episodes:
                    break
                ep_dir = root / 'episodes' / ep_id
                if not ep_dir.is_dir():
                    raise FileNotFoundError(f'Missing episode directory: {ep_dir}')
                ep_rows = rows_by_ep[ep_id]
                new_ep_id = f'{tag}__{ep_id}'
                tar_name = f'{new_ep_id}.tar'
                tar_path = out_dir / tar_name

                reused = False
                if resume and tar_name in existing_summary and tar_path.is_file():
                    rec = existing_summary[tar_name]
                    if tar_path.stat().st_size == rec.get('bytes') and sha256_file(tar_path) == rec.get('sha256'):
                        shard_records[tar_name] = rec
                        size = rec['bytes']
                        digest = rec['sha256']
                        reused = True
                        print(f'[pack] {tar_name}: resumed ({size / (1 << 20):.1f} MiB)', flush=True)

                if not reused:
                    tmp_path = out_dir / f'{tar_name}.tmp'
                    arc_prefix = f'episodes/{new_ep_id}'
                    with tarfile.open(tmp_path, mode='w') as tf:
                        if image_codec == 'png':
                            for path in sorted(ep_dir.rglob('*')):
                                rel = path.relative_to(ep_dir).as_posix()
                                tf.add(path, arcname=f'{arc_prefix}/{rel}', recursive=False)
                        elif image_codec == 'webp_lossless':
                            frame_dirs = []
                            for path in sorted(ep_dir.iterdir()):
                                if path.is_dir():
                                    frame_dirs.append(path)
                                    # Add empty dir and non-RGB files first
                                    for sub in sorted(path.iterdir()):
                                        if sub.name not in RGB_NAMES:
                                            rel = sub.relative_to(ep_dir).as_posix()
                                            tf.add(sub, arcname=f'{arc_prefix}/{rel}', recursive=False)
                                else:
                                    tf.add(path, arcname=f'{arc_prefix}/{path.name}', recursive=False)
                            futures = [
                                pool.submit(_encode_frame_webp, fd, (not verified_lossless and i == 0))
                                for i, fd in enumerate(frame_dirs)
                            ]
                            for fut in futures:
                                for webp_rel, raw, mtime in fut.result():
                                    info = tarfile.TarInfo(name=f'{arc_prefix}/{webp_rel}')
                                    info.size = len(raw)
                                    info.mtime = int(mtime)
                                    info.mode = 0o644
                                    tf.addfile(info, io.BytesIO(raw))
                            if frame_dirs and not verified_lossless:
                                verified_lossless = True
                                print('[pack] verified bit-exact lossless WebP (max_abs_diff == 0)', flush=True)
                        else:
                            raise ValueError(f'Unsupported image_codec: {image_codec}')
                    tmp_path.replace(tar_path)

                    digest = sha256_file(tar_path)
                    size = tar_path.stat().st_size
                    shard_records[tar_name] = {
                        'episode': new_ep_id,
                        'source_corpus': tag,
                        'source_episode': ep_id,
                        'split': ep_rows[0]['split'],
                        'map': ep_rows[0]['map'],
                        'frames': len(ep_rows),
                        'bytes': size,
                        'sha256': digest,
                    }
                    print(f'[pack] {tar_name}: {len(ep_rows)} frames, {size / (1 << 20):.1f} MiB, sha256={digest[:12]}', flush=True)

                total_bytes += size
                total_frames += len(ep_rows)
                for row in ep_rows:
                    copied = dict(row)
                    copied['episode'] = new_ep_id
                    copied['source_corpus'] = tag
                    copied['source_episode'] = ep_id
                    manifest_lines.append(json.dumps(copied, ensure_ascii=False))
                packed_eps += 1
            if limit_episodes is not None and packed_eps >= limit_episodes:
                break

    out_manifest = out_dir / 'manifest.jsonl'
    out_manifest.write_text('\n'.join(manifest_lines) + ('\n' if manifest_lines else ''), encoding='utf-8')
    manifest_digest = sha256_file(out_manifest)
    summary = {
        'image_codec': image_codec,
        'episode_count': packed_eps,
        'total_frames': total_frames,
        'total_bytes': total_bytes,
        'manifest_sha256': manifest_digest,
        'files': shard_records,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'[pack] done: {packed_eps} episodes, {total_frames} frames, {total_bytes / (1 << 30):.2f} GiB ({image_codec})', flush=True)
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--roots', nargs='+', required=True, help='Corpus roots (path or tag=path)')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--limit-episodes', type=int, default=None)
    p.add_argument('--image-codec', choices=('png', 'webp_lossless'), default='webp_lossless')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    pack(args.roots, args.out, args.limit_episodes, args.image_codec, args.workers, args.resume)
