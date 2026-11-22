"""Create and independently verify Stage 3.1 training shards."""
import argparse
import hashlib
import io
import json
import re
import shutil
import tarfile
import tempfile
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

import cv2
import numpy as np

from .geometry import CONFIG_SHA256, NAMES
from .qc_v3 import LABEL_SHAPES, episode_index, load_npz, read_image, run_qc
from .sim import require_collection_map

FORMAT = "bev-wam-x-stage31-v1"
MAX_BYTES = 1_500_000_000
BIT_PACK = {"lanes", "occupancy", "future_occupancy"}
EP_PATTERN = re.compile(r"ep-[0-9]{6}")
TAR_PATTERN = re.compile(r"stage31-[0-9]{5}\.tar")


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def split_episodes(entries, validation_fraction=.20, seed=317):
    """Take transitive closure under either route equality or initial-state equality."""
    if not 0 < validation_fraction < 1:
        raise ValueError("Validation fraction must be between zero and one")
    parents = list(range(len(entries)))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    seen = {}
    for i, entry in enumerate(entries):
        for key in ("route_id", "initial_state_id"):
            value = (key, entry[key])
            parents[root(i)] = root(seen.setdefault(value, i))
    components = defaultdict(list)
    for i, entry in enumerate(entries):
        components[root(i)].append(entry)
    groups = list(components.values())
    if len(groups) < 2:
        raise ValueError("Need at least two independent route/initial-state components")
    np.random.default_rng(seed).shuffle(groups)
    goal = sum(len(e["frame_ids"]) for e in entries) * validation_fraction
    lengths = np.cumsum([sum(len(e["frame_ids"]) for e in g) for g in groups])[:-1]
    cut = int(np.argmin(abs(lengths - goal))) + 1
    splits = {name: sorted(e["id"] for g in chunk for e in g)
              for name, chunk in (("val", groups[:cut]), ("train", groups[cut:]))}
    assert_split(entries, splits)
    return splits


def assert_split(entries, splits):
    by_id = {e["id"]: e for e in entries}
    train, val = (set(splits[key]) for key in ("train", "val"))
    if (len(by_id) != len(entries) or not train or not val or train & val or train | val != set(by_id) or
            len(train) != len(splits["train"]) or len(val) != len(splits["val"])):
        raise ValueError("Invalid train/val episode index")
    for key in ("route_id", "initial_state_id"):
        if {by_id[i][key] for i in train} & {by_id[i][key] for i in val}:
            raise ValueError(f"Train/val {key} leakage")
    for e in entries:
        town = require_collection_map(e["map"])
        if not EP_PATTERN.fullmatch(e["id"]) or any(
            not e[key].startswith(f"{town}/") or len(e[key].split("/")) != 2
            for key in ("route_id", "initial_state_id")
        ):
            raise ValueError("Invalid map/episode grouping identifier")
        ids = e["frame_ids"]
        if len(ids) < 4 or any(type(i) is not int or i < 0 for i in ids) or any(b != a + 1 for a, b in zip(ids, ids[1:])):
            raise ValueError("Invalid frame index")


def read_manifest(path, check_shards=True):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["format"] != FORMAT or manifest["config_sha256"] != CONFIG_SHA256:
        raise ValueError("Incompatible manifest format or sensor SHA-256")
    entries, shards = manifest["episodes"], manifest["shards"]
    assert_split(entries, manifest["splits"])
    listed = {s["name"] for s in shards}
    if (manifest["total_frames"] != sum(len(e["frame_ids"]) for e in entries) or
            listed != {e["shard"] for e in entries} or len(listed) != len(shards)):
        raise ValueError("Manifest frame/shard index mismatch")
    for shard in shards:
        name, size = shard["name"], shard["bytes"]
        if not TAR_PATTERN.fullmatch(name) or not 0 < size <= MAX_BYTES:
            raise ValueError(f"Invalid shard record: {shard}")
        if check_shards:
            destination = path.parent / name
            if destination.stat().st_size != size or sha256(destination) != shard["sha256"]:
                raise ValueError(f"Shard integrity failure: {name}")
    return manifest


def array_bytes(values):
    stream = io.BytesIO()
    np.savez_compressed(stream, **values)
    return stream.getvalue()


def bit_pack(array):
    return np.packbits(array.reshape(-1), bitorder="little")


def unpack_targets(arrays, index):
    return {key: (np.unpackbits(arrays[key][index], count=int(np.prod(shape)), bitorder="little").reshape(shape)
                  if key in BIT_PACK else arrays[key][index])
            for key, shape in LABEL_SHAPES.items()}


def episode_members(folder, entry):
    collected = []
    for fid in entry["frame_ids"]:
        path = folder / str(fid)
        for name in NAMES:
            rgb = read_image(path / f"{name}.png")
            ok, encoded = cv2.imencode(".webp", rgb, [cv2.IMWRITE_WEBP_QUALITY, 90])
            if not ok:
                raise IOError(f"WebP failed: {path}/{name}")
            yield f"{fid}/{name}.webp", encoded.tobytes()
        for name in ("obs", "audit", "depth"):
            yield f"{fid}/{name}.npz", array_bytes(load_npz(path / f"{name}.npz"))
        labels = load_npz(path / "labels.npz")
        collected.append({k: bit_pack(v) if k in BIT_PACK else v for k, v in labels.items()})
    yield "targets.npz", array_bytes({key: np.stack([v[key] for v in collected]) for key in LABEL_SHAPES})


def pack(source, out, max_bytes=MAX_BYTES, validation_fraction=.2, seed=317):
    if not 20_480 <= max_bytes <= MAX_BYTES:
        raise ValueError(f"Shard budget must be at most {MAX_BYTES:,} bytes")
    raw = sorted(episode_index(source), key=lambda pair: (str(pair[0].parent.resolve()), pair[0].name))
    entries = [dict(info, id=f"ep-{i:06d}") for i, (_, info) in enumerate(raw)]
    splits = split_episodes(entries, validation_fraction, seed)
    out.mkdir(parents=True, exist_ok=False)
    run_qc(source, out / "qc")
    paths = []
    active = None
    used = 0
    try:
        for (folder, _), entry in zip(raw, entries, strict=True):
            # Stage one episode on disk so preparation memory stays bounded and shards stay atomic.
            spool_path = out / ".episode.part"
            with tarfile.open(spool_path, "w", format=tarfile.USTAR_FORMAT) as spool:
                for name, content in episode_members(folder, entry):
                    member = tarfile.TarInfo(f"{entry['id']}/{name}")
                    member.size = len(content)
                    spool.addfile(member, io.BytesIO(content))
                episode_bytes = spool.offset
            if episode_bytes + 10240 > max_bytes:
                raise ValueError(f"Episode {folder} exceeds shard budget; collect shorter episodes")
            if active is None or used + episode_bytes + 10240 > max_bytes:
                if active is not None:
                    active.close()
                paths.append(out / f"stage31-{len(paths):05d}.tar")
                active = tarfile.open(paths[-1], "w", format=tarfile.USTAR_FORMAT)
                used = 0
            entry["shard"] = paths[-1].name
            with tarfile.open(spool_path, "r:") as spool:
                for member in spool:
                    with spool.extractfile(member) as stream:
                        active.addfile(member, stream)
            used += episode_bytes
            spool_path.unlink()
    finally:
        if active is not None:
            active.close()
        (out / ".episode.part").unlink(missing_ok=True)
    shards = [dict(name=p.name, sha256=sha256(p), bytes=p.stat().st_size) for p in paths]
    if any(s["bytes"] > max_bytes for s in shards):
        raise RuntimeError("Tar padding exceeded shard size budget")
    manifest = dict(format=FORMAT, config_sha256=CONFIG_SHA256, total_frames=sum(len(e["frame_ids"]) for e in entries),
                    splits=splits, episodes=entries, shards=shards, webp_quality=90, bit_order="little")
    temporary = out / "manifest.json.part"
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(out / "manifest.json")
    return manifest


def expected_members(entries):
    names = {}
    for entry in entries:
        prefix = entry["id"]
        names[f"{prefix}/targets.npz"] = entry["shard"]
        for frame in entry["frame_ids"]:
            for file in (*[f"{n}.webp" for n in NAMES], "obs.npz", "audit.npz", "depth.npz"):
                names[f"{prefix}/{frame}/{file}"] = entry["shard"]
    return names


def unpack_verified(manifest_path, out):
    manifest = read_manifest(manifest_path)
    expected = expected_members(manifest["episodes"])
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    with tempfile.TemporaryDirectory(prefix=".stage31-unpack-", dir=out.parent) as temp:
        stage = Path(temp) / "dataset"
        for shard in manifest["shards"]:
            with tarfile.open(manifest_path.parent / shard["name"], "r:") as archive:
                for member in archive:
                    if not member.isfile() or expected.pop(member.name, None) != shard["name"]:
                        raise ValueError(f"Unexpected or duplicate archive member: {member.name}")
                    dest = stage / member.name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with archive.extractfile(member) as stream, dest.open("xb") as output:
                        shutil.copyfileobj(stream, output)
        if expected:
            raise ValueError("Archive members missing")
        shutil.copyfile(manifest_path, stage / "manifest.json")
        stage.rename(out)
    return manifest


def verify_pack(source, manifest_path, samples=24, seed=314):
    manifest = read_manifest(manifest_path)
    raw = sorted(episode_index(source), key=lambda pair: (str(pair[0].parent.resolve()), pair[0].name))
    entries = manifest["episodes"]
    if len(raw) != len(entries) or samples < 1:
        raise ValueError("Original episode count or sample count invalid")
    rng = np.random.default_rng(seed)
    total = 0

    def verify_arrays(a, b):
        if set(a) != set(b):
            raise ValueError("Archive key mismatch")
        for key in a:
            if a[key].dtype != b[key].dtype or not np.array_equal(a[key], b[key]):
                raise ValueError(f"Exact non-RGB pack comparison failed: {key}")

    with ExitStack() as context:
        shards = {}
        for (folder, info), entry in zip(raw, entries, strict=True):
            if any(entry.get(k) != v for k, v in info.items()):
                raise ValueError("Original episode metadata changed")
            shard = entry["shard"]
            if shard not in shards:
                shards[shard] = context.enter_context(tarfile.open(manifest_path.parent / shard, "r:"))
            archive = shards[shard]
            prefix = entry["id"]

            def arrays(name):
                with archive.extractfile(f"{prefix}/{name}") as obj:
                    with np.load(io.BytesIO(obj.read()), allow_pickle=False) as npz:
                        return {k: npz[k] for k in npz.files}

            targets = arrays("targets.npz")
            for idx in rng.choice(len(entry["frame_ids"]), min(samples, len(entry["frame_ids"])),
                                  replace=False):
                fid = entry["frame_ids"][idx]
                src = folder / str(fid)
                original = load_npz(src / "labels.npz")
                recovered = unpack_targets(targets, idx)
                for k in BIT_PACK:
                    if original[k].dtype == np.bool_:
                        recovered[k] = recovered[k].astype(bool)
                verify_arrays(original, recovered)
                for name in ("obs", "audit", "depth"):
                    verify_arrays(load_npz(src / f"{name}.npz"), arrays(f"{fid}/{name}.npz"))
                for name in NAMES:
                    with archive.extractfile(f"{prefix}/{fid}/{name}.webp") as fr:
                        img = cv2.imdecode(np.frombuffer(fr.read(), np.uint8), cv2.IMREAD_COLOR)
                    if img is None or img.shape != read_image(src / f"{name}.png").shape:
                        raise ValueError(f"WebP shape mismatch: {src}/{name}")
                total += 1
    return dict(sampled_frames=total, max_abs_diff=0.0, route_overlap=0, initial_state_overlap=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-shard-bytes", type=int, default=MAX_BYTES)
    parser.add_argument("--verify-pack", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--unpack", type=Path, help="Manifest to verify and extract into --out")
    args = parser.parse_args()
    if args.unpack:
        unpack_verified(args.unpack, args.out)
        return
    if not args.source:
        parser.error("--source is required to pack or compare")
    if not args.verify_only:
        pack(args.source, args.out, max_bytes=args.max_shard_bytes)
    if args.verify_pack or args.verify_only:
        result = verify_pack(args.source, args.out / "manifest.json")
        (args.out / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(result)


if __name__ == "__main__":
    main()
