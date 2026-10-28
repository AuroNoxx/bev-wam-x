"""End-to-end pipeline for Stage 7 risk data collection, feature caching,
world/zero training, Full/Zero closed-loop evaluation, and pilot reporting."""
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
from mini_wa_jepa.provenance import provenance, write_json


def main():
    root = Path('artifacts/stage7_integrated')
    status_file = root / 'risk_pipeline_014.json'
    if status_file.exists():
        raise FileExistsError(f"Status file already exists: {status_file}")

    report = dict(
        **provenance(),
        completed=False,
        stage7_acceptance=False,
        stage8_started=False,
        phase='initializing',
        commands=[],
    )
    write_json(status_file, report)

    def run_step(name, args, log_name=None):
        report['phase'] = name
        write_json(status_file, report)
        cmd = [sys.executable, '-m', *args]
        log_file = root / (log_name or f'{name}.log')
        print(f"=== Starting step: {name} ===", flush=True)
        with log_file.open('w') as stream:
            subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT, check=True)
        print(f"=== Completed step: {name} ===", flush=True)
        report['commands'].append(dict(name=name, arguments=args, log=str(log_file), success=True))
        write_json(status_file, report)

    flow_ckpt = 'artifacts/stage7_integrated/mixed_flow_010/flow_20261023/best.pt'
    mixed_val = 'artifacts/stage7_integrated/mixed_validation_010'

    try:
        # Step 1: Collect risk train (32 contexts, 384 branches)
        run_step(
            'collect_risk_train',
            [
                'scripts.collect_integrated_risk',
                '--flow', flow_ckpt,
                '--split', 'train',
                '--context-profile', 'near_single',
                '--reuse-engine',
                '--output', str(root / 'risk_train_014'),
            ],
            log_name='risk_train_014.log',
        )

        # Step 2: Collect risk validation (16 contexts, 192 branches)
        run_step(
            'collect_risk_validation',
            [
                'scripts.collect_integrated_risk',
                '--flow', flow_ckpt,
                '--split', 'validation',
                '--context-profile', 'near_single',
                '--reuse-engine',
                '--output', str(root / 'risk_validation_014'),
            ],
            log_name='risk_validation_014.log',
        )

        # Step 3: Cache WAM features
        run_step(
            'cache_risk_features',
            [
                'scripts.cache_integrated_risk',
                '--sources',
                str(root / 'risk_train_014'),
                str(root / 'risk_validation_014'),
                '--output',
                str(root / 'risk_features_014'),
            ],
            log_name='risk_features_014.log',
        )

        # Step 4: Train matched world/zero risk heads
        run_step(
            'train_risk_heads',
            [
                'scripts.train_integrated_risk',
                '--data', str(root / 'risk_features_014'),
                '--output', str(root / 'risk_runs_014'),
            ],
            log_name='risk_runs_014.log',
        )

        # Step 5: Evaluate Zero arm (pilot10)
        run_step(
            'evaluate_zero_closed',
            [
                'scripts.evaluate_integrated_driving',
                '--arm', 'zero',
                '--checkpoint', flow_ckpt,
                '--risk', str(root / 'risk_runs_014/zero_20261023/calibrated.pt'),
                '--validation', mixed_val,
                '--output', str(root / 'mixed_zero_closed_014'),
            ],
            log_name='mixed_zero_closed_014.log',
        )

        # Step 6: Evaluate Full arm (pilot10)
        run_step(
            'evaluate_full_closed',
            [
                'scripts.evaluate_integrated_driving',
                '--arm', 'full',
                '--checkpoint', flow_ckpt,
                '--risk', str(root / 'risk_runs_014/world_20261023/calibrated.pt'),
                '--validation', mixed_val,
                '--output', str(root / 'mixed_full_closed_014'),
            ],
            log_name='mixed_full_closed_014.log',
        )

        # Step 7: Audit all 5 arms
        run_step(
            'audit_5_arms',
            [
                'scripts.report_integrated_pilot',
                '--sources',
                str(root / 'mixed_joint_closed_012'),
                str(root / 'mixed_resnet_closed_012'),
                str(root / 'mixed_geometry_closed_012'),
                str(root / 'mixed_zero_closed_014'),
                str(root / 'mixed_full_closed_014'),
                '--output',
                str(root / 'mixed_driving_audit_014'),
            ],
            log_name='mixed_driving_audit_014.log',
        )

        report.update(completed=True, phase='all_steps_completed_not_stage7_acceptance')
        print("=== Full pipeline 014 completed successfully ===", flush=True)
    except BaseException:
        err = traceback.format_exc()
        report.update(phase='failed_preserved', error=err)
        print(f"Pipeline failed: {err}", file=sys.stderr, flush=True)
        raise
    finally:
        write_json(status_file, report)


if __name__ == '__main__':
    main()
