"""Audit schema3 navigation labels and native-event provenance without rerunning sim."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from dataset.recording import future_labels_v2
from scripts.collect_integrated_navigation import digest
from mini_wa_jepa.provenance import write_json


def audit(root, output):
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((root/'manifest.json').read_text())
    if not manifest.get('control_verified') or (root/'INVALID_CONTROL_WIRING.json').exists():
        raise ValueError('Unverified/invalid ego control: data is not valid for driving claims or training')
    if not manifest['completed']:
        raise ValueError('Cannot audit incomplete collection as complete')
    rows = []
    for episode in manifest['episodes']:
        raw_path = root/episode['raw']
        assert digest(raw_path) == episode['raw_sha256']
        with np.load(raw_path, allow_pickle=False) as raw:
            np.testing.assert_allclose(raw['action'], raw['applied_action'], rtol=0, atol=1e-7)
            n = episode['steps']; assert len(raw['states']) == n+1
            target, valid, _ = future_labels_v2(raw['states'], np.zeros(n+1, bool), 10, 2)
            np.testing.assert_array_equal(raw['valid'], valid[:n])
            np.testing.assert_array_equal(raw['trajectory'], target[:n])
            assert raw['bev'].shape == (n, 4, 128, 128) and raw['bev'].dtype == np.uint8
            assert raw['route_points'].shape == (n, 8, 2)
            assert np.isfinite(raw['route_points']).all()
            assert not np.any(raw['route_points'][~raw['route_valid']])
            assert np.all(raw['history_index'] <= np.arange(n)[:, None])
            assert np.all(raw['history_index'] >= 0)
            np.testing.assert_array_equal(raw['history_index'][:, -1], np.arange(n))
            assert abs(raw['reward'].sum()-episode['reward']) < 1e-8
            assert abs(raw['cost'].sum()-episode['cost']) < 1e-8
            expected = np.array([episode['outcome'][k] for k in manifest['events']])
            np.testing.assert_array_equal(raw['events'][-1], expected)
            if manifest.get('input_profile') == 'schema3.1_static_maneuver_permissions':
                assert raw['maneuver_allowed'].shape == (n, 2) and raw['maneuver_allowed'].dtype == bool
                if 'diagnostic_npc_states' in raw:
                    assert raw['bev'][0, 2].any()
                    assert len(raw['diagnostic_npc_states']) == n
                    if episode.get('maneuver_demonstration_success'):
                        assert raw['diagnostic_npc_present_after'].all()
                        assert not raw['events'][:, [0, 4]].any()
                        phases = raw['phase'].tolist()
                        if episode['case'].startswith('allowed'):
                            sequence = [phases.index(k) for k in ('FOLLOW', 'CHANGE_OUT', 'PASS', 'CHANGE_BACK', 'COMPLETE')]
                            assert sequence == sorted(sequence)
                            assert phases.count('COMPLETE') >= 31
                            side = raw['lateral_m'].min() if episode['case']=='allowed_left' else raw['lateral_m'].max()
                            assert side < -3 if episode['case']=='allowed_left' else side > 3
                            assert abs(raw['lateral_m'][-1]) < .35
                        else:
                            assert set(phases) == {'FOLLOW'}
                            assert np.max(np.abs(raw['lateral_m'])) < .35
        rrd = raw_path.parent/'navigation.rrd'
        assert digest(rrd) == episode['rerun_sha256']
        with (output/f"seed_{episode['seed']}_rrd.txt").open('w') as stream:
            result = subprocess.run([sys.executable, '-m', 'rerun', 'rrd', 'print', str(rrd)],
                                    stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError('Rerun native readback failed')
        rows.append(dict(seed=episode['seed'], frames=n, native_rrd_readback=True,
                         raw_sha256=episode['raw_sha256']))
    report = dict(completed=True, source_manifest_sha256=digest(root/'manifest.json'),
                  schema_version=3, episodes=rows, total_frames=sum(r['frames'] for r in rows),
                  success_demonstrations=sum(e['success_demonstration'] for e in manifest['episodes']),
                  input_profile=manifest.get('input_profile', 'schema3_navigation'),
                  event_episodes={k:sum(e['outcome'][k] for e in manifest['episodes']) for k in manifest['events']})
    write_json(output/'audit.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); audit(args.root, args.output)
