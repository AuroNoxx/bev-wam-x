"""Bounded local pipeline for the registered mixed-data navigation pilots."""
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
from mini_wa_jepa.provenance import provenance, write_json


def main():
    root = Path('artifacts/stage7_integrated'); status = root/'mixed_learning_pipeline_010.json'
    if status.exists():
        raise FileExistsError(status)
    report = dict(**provenance(), completed=False, stage7_acceptance=False, stage8_started=False,
                  phase='waiting_for_registered_collections', commands=[])
    write_json(status, report)
    def run(name, arguments):
        report['phase'] = name; write_json(status, report)
        command = [sys.executable, '-m', *arguments]
        with (root/f'mixed010_{name}.log').open('x') as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
        report['commands'].append(dict(name=name, arguments=arguments, success=True)); write_json(status, report)
    try:
        sources = ['navigation_train_009', 'navigation_validation_009', 'traffic_train_009', 'traffic_validation_009']
        deadline = time.monotonic()+7200
        while True:
            rows = [json.loads((root/name/'manifest.json').read_text()) for name in sources]
            for row in rows:
                if row.get('error'):
                    raise RuntimeError('A registered source failed: '+row['error'])
            if all(row['completed'] for row in rows):
                break
            if time.monotonic() > deadline:
                raise TimeoutError('Registered collection exceeded bounded wait')
            time.sleep(10)
        for name in sources:
            run('audit_'+name, ['scripts.audit_integrated_navigation', '--root', str(root/name), '--output', str(root/name/'audit')])
        for split in ('train', 'validation'):
            run('merge_'+split, ['scripts.merge_integrated_navigation', '--sources', str(root/f'navigation_{split}_009'),
                str(root/f'traffic_{split}_009'), '--output', str(root/f'mixed_{split}_010')])
        for encoder in ('joint', 'resnet'):
            run('train_'+encoder, ['scripts.train_navigation_pilot', '--encoder', encoder, '--reference-prior',
                '--train', str(root/'mixed_train_010'), '--validation', str(root/'mixed_validation_010'),
                '--output', str(root/f'mixed_{encoder}_010')])
        run('train_flow', ['scripts.train_navigation_flow', '--cache', str(root/'mixed_joint_010'),
            '--output', str(root/'mixed_flow_010')])
        report.update(completed=True, phase='mixed_models_trained_not_acceptance')
    except BaseException:
        report.update(phase='failed_preserved', error=traceback.format_exc()); raise
    finally:
        write_json(status, report)


if __name__ == '__main__':
    main()
