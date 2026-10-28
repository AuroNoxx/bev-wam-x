"""Merge base mixed dataset with targeted supplement into unified Schema 3.1 dataset."""

import argparse
import json
from pathlib import Path
import sys

WORKSPACE = Path(__file__).resolve().parent.parent
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from mini_wa_jepa.provenance import provenance, write_json
from scripts.collect_integrated_navigation import digest, EVENTS


def merge(base_manifest_path, supplement_manifest_path, output_manifest_path, split="train"):
    base_manifest_path = Path(base_manifest_path)
    supplement_manifest_path = Path(supplement_manifest_path)
    output_manifest_path = Path(output_manifest_path)
    output_dir = output_manifest_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(base_manifest_path, "r") as f:
        base = json.load(f)
    with open(supplement_manifest_path, "r") as f:
        supp = json.load(f)

    if not base.get("control_verified") or not supp.get("control_verified"):
        raise ValueError("Both datasets must be control_verified")

    merged_episodes = []
    seen_seeds = set()

    for ep in base.get("episodes", []):
        seed = ep["seed"]
        if seed in seen_seeds:
            continue
        seen_seeds.add(seed)
        raw_path = Path(ep["raw"])
        if not raw_path.is_absolute():
            raw_path = (base_manifest_path.parent / raw_path).resolve()
        merged_episodes.append(dict(ep, raw=str(raw_path)))

    for ep in supp.get("episodes", []):
        seed = ep["seed"]
        if seed in seen_seeds:
            continue
        seen_seeds.add(seed)
        raw_path = Path(ep["raw"])
        if not raw_path.is_absolute():
            raw_path = (supplement_manifest_path.parent / raw_path).resolve()
        merged_episodes.append(dict(ep, raw=str(raw_path)))

    merged_manifest = dict(
        **provenance(),
        schema_version=3,
        input_profile="schema3.1_static_maneuver_permissions",
        split=split,
        completed=True,
        control_verified=True,
        native_termination=True,
        navigation_permission="map_assisted_route_lane_centerline",
        events=list(EVENTS),
        base_dataset=str(base_manifest_path),
        supplement_dataset=str(supplement_manifest_path),
        total_episodes=len(merged_episodes),
        episodes=merged_episodes,
    )

    write_json(output_manifest_path, merged_manifest)
    print(f"Successfully merged {len(merged_episodes)} episodes into {output_manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, required=True)
    parser.add_argument("--supplement", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    args = parser.parse_args()

    merge(args.base, args.supplement, args.output, split=args.split)
