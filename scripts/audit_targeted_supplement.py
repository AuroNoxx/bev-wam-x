"""Independent raw-NPZ stop/resume audit; never trusts manifest success flags.

Other scenarios remain explicitly not audited by this revision. Original NPZs
and manifests are read only; --output must name a fresh JSON file.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

WORKSPACE = Path(__file__).resolve().parent.parent
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))
from dataset.behavior_audit import STOP_CASES, audit_episode


def audit_supplement(manifest_path, output=None, stop_only=False):
    manifest_path = Path(manifest_path).resolve()
    data = json.loads(manifest_path.read_text())
    records = []
    for ep in data.get('episodes', []):
        if stop_only and ep['case_type'] not in STOP_CASES:
            continue
        raw = Path(ep['raw'])
        if not raw.is_absolute():
            raw = manifest_path.parent / raw
        try:
            digest = hashlib.sha256(raw.read_bytes()).hexdigest()
            with np.load(raw, allow_pickle=False) as arrays:
                result = audit_episode(arrays, ep['case_type'], data.get('events', []))
            if ep.get('raw_sha256') != digest:
                result['failures'].append('manifest_raw_hash_missing_or_mismatch')
                result['passed'] = False
        except (OSError, ValueError, KeyError, IndexError) as error:
            result = dict(passed=False, failures=['raw_audit_error: ' + str(error)], missing=[], metrics={})
            digest = None
        records.append(dict(seed=ep['seed'], case_type=ep['case_type'], raw=str(raw),
                            raw_sha256=digest, manifest_claim=ep.get('success_demonstration'), **result))
    passed = sum(r['passed'] for r in records)
    report = dict(manifest=str(manifest_path),
                  manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                  audit_source_sha256=hashlib.sha256((WORKSPACE/'dataset/behavior_audit.py').read_bytes()).hexdigest(),
                  scope='stop_resume_only' if stop_only else 'all_selected_cases',
                  episodes=records, total=len(records), passed=passed,
                  failed=len(records)-passed, all_passed=bool(records) and passed == len(records))
    if output:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x') as f:
            json.dump(report, f, indent=2)
            f.write('\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'episodes'}, indent=2))
    for r in records:
        print(r['seed'], r['case_type'], 'PASS' if r['passed'] else 'FAIL/MISSING',
              json.dumps(dict(metrics=r['metrics'], failures=r['failures'], missing=r['missing'])))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('manifest')
    parser.add_argument('--output')
    parser.add_argument('--stop-only', action='store_true')
    args = parser.parse_args()
    report = audit_supplement(args.manifest, args.output, args.stop_only)
    raise SystemExit(0 if report['all_passed'] else 1)
