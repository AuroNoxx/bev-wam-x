"""Bounded Stage7 recovery-pilot continuation; never touches scientific test/Stage8."""
from pathlib import Path
import json
import subprocess
import sys
import time
import traceback
from mini_wa_jepa.provenance import write_json, provenance


def run():
    root = Path('artifacts/stage7_integrated')
    status = root/'navigation_recovery_pipeline_003.json'
    if status.exists():
        raise FileExistsError(status)
    state = dict(**provenance(), completed=False, phase='waiting_for_registered_collection',
                 scope='Stage7 recovery development pilot only', stage8_started=False)
    def update(phase):
        state['phase'] = phase; write_json(status, state)
    def command(module, arguments, log_name):
        with (root/log_name).open('x') as log:
            subprocess.run([sys.executable, '-m', module, *arguments],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    update(state['phase'])
    try:
        deadline = time.monotonic()+2400
        sources = [root/'navigation_train_pilot_003', root/'navigation_validation_pilot_003']
        while True:
            manifests = [json.loads((source/'manifest.json').read_text()) for source in sources]
            for manifest in manifests:
                if manifest.get('error'):
                    raise RuntimeError(manifest['error'])
                if not manifest.get('control_verified'):
                    raise RuntimeError('Unverified external-control collection')
            if all(m['completed'] for m in manifests):
                break
            if time.monotonic() > deadline:
                raise TimeoutError('Registered collection exceeded40min bound')
            time.sleep(2)
        update('audit_recovery_data')
        for source in sources:
            command('scripts.audit_integrated_navigation', ['--root', str(source), '--output', str(source/'audit')],
                    source.name+'_audit.log')
        update('training_two_navigation_heads')
        command('scripts.train_navigation_pilot', ['--train', str(sources[0]), '--validation', str(sources[1]),
                '--output', str(root/'navigation_head_pilot_003')], 'navigation_head_pilot_003.log')
        update('closed_validation_six_registered_scenes')
        command('scripts.evaluate_navigation_pilot', ['--checkpoint', str(root/'navigation_head_pilot_003/head_20261023/best.pt'),
                '--validation', str(sources[1]), '--output', str(root/'navigation_closed_validation_003')],
                'navigation_closed_validation_003.log')
        update('audit_and_render_interim_progress')
        command('scripts.report_navigation_startup', ['--version', '003'], 'navigation_progress_003.log')
        state['completed'] = True; update('development_pilot_completed_not_stage_acceptance')
    except BaseException:
        state['error'] = traceback.format_exc(); update('failed_preserve_evidence'); raise


if __name__ == '__main__':
    run()
