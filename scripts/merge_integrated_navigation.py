"""Create a manifest-only union of audited compatible navigation/traffic data."""
import argparse
import json
from pathlib import Path
from mini_wa_jepa.provenance import provenance, write_json
from scripts.collect_integrated_navigation import digest


def merge(sources, output):
    output.mkdir(parents=True, exist_ok=False)
    episodes = []; scenes = []; origins = []; profile = None; split = None; seeds = set()
    for source in sources:
        manifest = json.loads((source/'manifest.json').read_text())
        audit = json.loads((source/'audit'/'audit.json').read_text())
        if (not manifest['completed'] or not manifest.get('control_verified') or
                (source/'INVALID_CONTROL_WIRING.json').exists()):
            raise ValueError('Only verified complete sources can be merged')
        if audit['source_manifest_sha256'] != digest(source/'manifest.json') or not audit['completed']:
            raise ValueError('Missing/current source audit required')
        if profile is None:
            profile = manifest['input_profile']; split = manifest['split']
        if manifest['input_profile'] != profile or manifest['split'] != split:
            raise ValueError('Cannot combine missing or different input profiles/splits')
        for ep in manifest['episodes']:
            if ep['seed'] in seeds:
                raise ValueError('Repeated episode seed in merged split')
            seeds.add(ep['seed'])
            path = (source/ep['raw']).resolve()
            if digest(path) != ep['raw_sha256']:
                raise ValueError('Raw checksum mismatch')
            episodes.append(dict(ep, raw=str(path), source_collection=str(source.resolve())))
        scenes.extend(manifest['scenes'])
        origins.append(dict(path=str(source.resolve()), manifest_sha256=digest(source/'manifest.json'),
                            audit_sha256=digest(source/'audit'/'audit.json')))
    write_json(output/'manifest.json', dict(**provenance(), completed=True, schema_version=3,
        input_profile=profile, split=split, control_verified=True, native_termination=True,
        navigation_permission='map_assisted_route_lane_centerline',
        source_collections=origins, scenes=scenes, episodes=episodes,
        events=manifest['events'], training_weight='equal_per_eligible_actual_anchor',
        future_actor_truth_for_model=False, test_access=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--sources', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True); a = p.parse_args(); merge(a.sources, a.output)
