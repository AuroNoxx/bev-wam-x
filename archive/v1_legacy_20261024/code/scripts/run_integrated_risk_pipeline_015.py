"""End-to-end pipeline for Stage 7 targeted risk head repair (Run 015).

Implements:
1. Balanced dataset with stationary launch (anchor 0) and stop-and-restart (anchor 50).
2. Separation of Flow candidates vs Diagnostic rules, reporting normal-scene false rejection rate.
3. Strict gate check before full closed loop (launch, resume, emergency brake).
4. 5-arm comparison report and video rendering.
"""
import json
from pathlib import Path
import subprocess
import sys
import traceback
from mini_wa_jepa.provenance import provenance, write_json


def main():
    root = Path('artifacts/stage7_integrated')
    status_file = root / 'risk_pipeline_015.json'
    if status_file.exists():
        raise FileExistsError(f"Status file already exists: {status_file}")

    report = dict(
        **provenance(),
        completed=False,
        stage7_acceptance=False,
        stage8_started=False,
        targeted_experiment='risk_deadlock_repair_015',
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
        # Step 1: Collect balanced risk training set (32 contexts, 50% normal / 50% hazard)
        run_step(
            'collect_risk_train',
            [
                'scripts.collect_integrated_risk',
                '--flow', flow_ckpt,
                '--split', 'train',
                '--context-profile', 'balanced_restart',
                '--reuse-engine',
                '--output', str(root / 'risk_train_015'),
            ],
            log_name='risk_train_015.log',
        )

        # Step 2: Collect balanced risk validation set (16 contexts)
        run_step(
            'collect_risk_validation',
            [
                'scripts.collect_integrated_risk',
                '--flow', flow_ckpt,
                '--split', 'validation',
                '--context-profile', 'balanced_restart',
                '--reuse-engine',
                '--output', str(root / 'risk_validation_015'),
            ],
            log_name='risk_validation_015.log',
        )

        # Step 3: Cache WAM features
        run_step(
            'cache_risk_features',
            [
                'scripts.cache_integrated_risk',
                '--sources',
                str(root / 'risk_train_015'),
                str(root / 'risk_validation_015'),
                '--output',
                str(root / 'risk_features_015'),
            ],
            log_name='risk_features_015.log',
        )

        # Step 4: Train matched world/zero risk heads
        run_step(
            'train_risk_heads',
            [
                'scripts.train_integrated_risk',
                '--data', str(root / 'risk_features_015'),
                '--output', str(root / 'risk_runs_015'),
            ],
            log_name='risk_runs_015.log',
        )

        # Step 5: Gate short-check (Open-road launch, stop-and-resume, hazardous braking)
        world_risk_ckpt = str(root / 'risk_runs_015/world_20261023/calibrated.pt')
        print("=== Starting Gate Short-Check ===", flush=True)
        report['phase'] = 'gate_short_check'
        write_json(status_file, report)
        gate_proc = subprocess.run(
            [sys.executable, '-m', 'scripts.gate_check_risk_head', '--flow', flow_ckpt,
             '--risk', world_risk_ckpt, '--output', str(root / 'risk_runs_015')],
            capture_output=True, text=True
        )
        gate_log = root / 'gate_check_015.log'
        gate_log.write_text(gate_proc.stdout + '\n' + gate_proc.stderr)
        if gate_proc.returncode != 0:
            print("!!! GATE SHORT-CHECK FAILED - ABORTING FULL CLOSED-LOOP EVALUATION !!!", flush=True)
            print(gate_proc.stdout, flush=True)
            report.update(
                phase='gate_check_failed_preserved',
                error='Gate check failed: risk head did not pass launch/resume/braking short-tests.',
                gate_check_output=gate_proc.stdout
            )
            write_json(status_file, report)
            return

        print("=== GATE SHORT-CHECK PASSED! Proceeding to full closed-loop evaluations ===", flush=True)
        report['commands'].append(dict(name='gate_short_check', success=True, log=str(gate_log)))
        write_json(status_file, report)

        # Step 6: Evaluate Zero arm (pilot10)
        run_step(
            'evaluate_zero_closed',
            [
                'scripts.evaluate_integrated_driving',
                '--arm', 'zero',
                '--checkpoint', flow_ckpt,
                '--risk', str(root / 'risk_runs_015/zero_20261023/calibrated.pt'),
                '--validation', mixed_val,
                '--output', str(root / 'mixed_zero_closed_015'),
            ],
            log_name='mixed_zero_closed_015.log',
        )

        # Step 7: Evaluate Full arm (pilot10)
        run_step(
            'evaluate_full_closed',
            [
                'scripts.evaluate_integrated_driving',
                '--arm', 'full',
                '--checkpoint', flow_ckpt,
                '--risk', world_risk_ckpt,
                '--validation', mixed_val,
                '--output', str(root / 'mixed_full_closed_015'),
            ],
            log_name='mixed_full_closed_015.log',
        )

        # Step 8: Audit all 5 arms with multi-arm video rendering
        run_step(
            'audit_5_arms',
            [
                'scripts.report_integrated_pilot',
                '--sources',
                str(root / 'mixed_joint_closed_012'),
                str(root / 'mixed_resnet_closed_012'),
                str(root / 'mixed_geometry_closed_012'),
                str(root / 'mixed_zero_closed_015'),
                str(root / 'mixed_full_closed_015'),
                '--output',
                str(root / 'mixed_driving_audit_015'),
            ],
            log_name='mixed_driving_audit_015.log',
        )

        report.update(completed=True, phase='all_steps_completed_targeted_repair_not_stage7_acceptance')
        print("=== Targeted repair pipeline 015 completed successfully ===", flush=True)
    except BaseException:
        err = traceback.format_exc()
        report.update(phase='failed_preserved', error=err)
        print(f"Pipeline failed: {err}", file=sys.stderr, flush=True)
        raise
    finally:
        write_json(status_file, report)


if __name__ == '__main__':
    main()
