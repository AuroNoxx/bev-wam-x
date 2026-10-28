"""Phase 0 Audit and Verification for BEV WA-JEPA Method Alignment.

Inspects source trees, git commit, dirty status, process table, hardware resources,
datasets, demonstration qualifications, cleanup candidates, and resource calibration plan.
Strictly read-only; performs dry-run audits without deleting any files.
"""

import os
import sys
import json
import glob
import psutil
import shutil
import hashlib
import subprocess
from pathlib import Path
import numpy as np
import torch

WORKSPACE = Path("/home/auronox/Desktop/mini-wa-jepa")
ARTIFACT_DIR = WORKSPACE / "artifacts/bev_wa_alignment/phase0_001"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def sha256_file(path):
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def audit_system_and_hardware():
    vram_bytes = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
    vm = psutil.virtual_memory()
    disk = shutil.disk_usage(str(WORKSPACE))
    
    # Active python processes
    python_procs = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'username']):
        try:
            cmd = proc.info.get('cmdline') or []
            cmd_str = " ".join(cmd)
            if 'python' in proc.info.get('name', '').lower() or any('python' in arg.lower() for arg in cmd):
                # Filter out our own process
                if proc.pid != os.getpid():
                    python_procs.append({
                        'pid': proc.info['pid'],
                        'user': proc.info['username'],
                        'cmd': cmd_str[:120]
                    })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    return {
        "gpu": {
            "device_name": gpu_name,
            "vram_total_bytes": vram_bytes,
            "vram_total_gb": round(vram_bytes / (1024**3), 4),
            "backend": "ROCm / HIP" if "rocm" in torch.__version__ else "CUDA/CPU",
            "torch_version": torch.__version__,
            "sdpa_available": hasattr(torch.nn.functional, "scaled_dot_product_attention")
        },
        "cpu": {
            "logical_cores": psutil.cpu_count(logical=True),
            "physical_cores": psutil.cpu_count(logical=False),
            "ram_total_gb": round(vm.total / (1024**3), 4),
            "ram_available_gb": round(vm.available / (1024**3), 4)
        },
        "disk": {
            "total_gb": round(disk.total / (1024**3), 4),
            "free_gb": round(disk.free / (1024**3), 4),
            "used_gb": round(disk.used / (1024**3), 4)
        },
        "active_python_processes": python_procs
    }


def audit_git_and_provenance():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=WORKSPACE).decode().strip()
    status_output = subprocess.check_output(["git", "status", "--porcelain"], cwd=WORKSPACE).decode()
    is_dirty = len(status_output.strip()) > 0
    
    tracked_sources = [
        "AGENTS.md", "GOAL.md", "TASKS.md", "README.md",
        "docs/legacy/GOAL_before_bev_wa_jepa_alignment_20261022.md",
        "docs/legacy/AGENTS_before_bev_wa_jepa_alignment_20261022.md",
        "docs/BEV_WA_JEPA_ALIGNMENT_REPORT.md",
        "docs/BEV_WA_JEPA_CLEANUP_GUIDE.md",
        "docs/GEMINI_BEV_WA_JEPA_EXECUTION_PROMPT.md",
        "models/latent.py", "models/joint.py", "models/action_conditioned.py",
        "models/navigation_flow.py", "models/integrated_world.py",
        "models/navigation_encoder.py", "models/navigation_head.py",
        "dataset/joint.py", "dataset/latent.py", "dataset/navigation.py",
        "dataset/risk_features.py", "dataset/flow_features.py",
        "training/joint.py", "training/flow_matching.py", "training/joint_loss.py",
        "planning/integrated.py", "planning/integrated_policy.py",
        "env/navigation.py", "env/integrated.py"
    ]
    
    file_hashes = {}
    for f in tracked_sources:
        p = WORKSPACE / f
        if p.exists():
            file_hashes[f] = sha256_file(p)
            
    return {
        "git_commit": commit,
        "is_dirty": is_dirty,
        "dirty_file_count": len(status_output.strip().splitlines()) if is_dirty else 0,
        "file_hashes": file_hashes
    }


def audit_datasets():
    # Audit mixed_train_010 and mixed_validation_010
    datasets = {
        "mixed_train_010": WORKSPACE / "artifacts/stage7_integrated/mixed_train_010/manifest.json",
        "mixed_validation_010": WORKSPACE / "artifacts/stage7_integrated/mixed_validation_010/manifest.json",
        "navigation_train_009": WORKSPACE / "artifacts/stage7_integrated/navigation_train_009/manifest.json",
        "traffic_train_009": WORKSPACE / "artifacts/stage7_integrated/traffic_train_009/manifest.json",
        "stage2_v2_100k": WORKSPACE / "outputs/data/stage2_v2_100k/manifest.json"
    }
    
    data_audit = {}
    for name, manifest_path in datasets.items():
        if not manifest_path.exists():
            data_audit[name] = {"status": "missing"}
            continue
            
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
            
        episodes = manifest.get("episodes", [])
        total_steps = sum(ep.get("steps", 0) for ep in episodes)
        seeds = [ep["seed"] for ep in episodes if "seed" in ep]
        
        # Analyze demonstration qualification & future availability on stage 7 datasets
        eligible_action_anchors = 0
        anchors_with_all_4_future_bevs = 0
        total_anchors = 0
        
        if name in ("mixed_train_010", "mixed_validation_010"):
            for ep in episodes:
                raw_path = Path(ep.get("raw", ""))
                if raw_path.exists():
                    try:
                        d = np.load(raw_path)
                        n_bev = len(d['bev'])
                        total_anchors += n_bev
                        
                        # 4 future BEVs at +5, +10, +15, +20 (covering 2.0s)
                        for t in range(n_bev):
                            if t + 20 < n_bev:
                                anchors_with_all_4_future_bevs += 1
                                
                        # Action eligibility check
                        valid = d.get('valid', np.zeros((n_bev, 10), bool))
                        route_valid = d.get('route_valid', np.zeros((n_bev, 8), bool))
                        events = d.get('events', np.zeros((n_bev, 7), bool))
                        noise_active = d.get('noise_active', None)
                        
                        bad = events[:, 0].astype(bool) | events[:, 4].astype(bool)
                        if noise_active is not None:
                            bad |= np.asarray(noise_active, bool)
                        prefix = np.r_[0, np.cumsum(bad)]
                        steps = np.arange(n_bev)
                        invalid_future = prefix[np.minimum(steps + 20, n_bev)] - prefix[steps]
                        eligible = valid.all(1) & (route_valid.sum(1) >= 2) & (invalid_future == 0)
                        eligible_action_anchors += int(eligible.sum())
                    except Exception as e:
                        pass
                        
        data_audit[name] = {
            "status": "valid",
            "episodes_count": len(episodes),
            "total_steps": total_steps,
            "seed_range": [min(seeds), max(seeds)] if seeds else [],
            "seed_count": len(set(seeds)),
            "schema_version": manifest.get("schema_version"),
            "input_profile": manifest.get("input_profile"),
            "control_verified": manifest.get("control_verified", False),
            "total_anchors": total_anchors if total_anchors > 0 else total_steps,
            "anchors_with_all_4_future_bevs": anchors_with_all_4_future_bevs,
            "eligible_action_anchors": eligible_action_anchors
        }
        
    return data_audit


def audit_cleanup_candidates():
    candidates = [
        {
            "name": "stage2_baseline_cache",
            "path": WORKSPACE / "outputs/data/stage2_v2_100k/baseline_cache",
            "type": "directory",
            "regenerator": "dataset/baseline.py:build_cache",
            "precondition": "outputs/data/stage2_v2_100k/{train,val,test} raw npz and manifest intact"
        },
        {
            "name": "flow_features_001",
            "path": WORKSPACE / "artifacts/restart_stage7/flow_features_001",
            "type": "directory",
            "regenerator": "scripts/cache_integrated_flow.py or equivalent token extraction",
            "precondition": "Stage 4 frozen encoder, raw data, and manifest intact"
        },
        {
            "name": "navigation_head_pilot_002_tokens",
            "path": WORKSPACE / "artifacts/stage7_integrated/navigation_head_pilot_002/train/tokens.npy",
            "type": "file",
            "regenerator": "scripts/train_navigation_pilot.py cache generator",
            "precondition": "raw navigation data intact"
        },
        {
            "name": "navigation_head_pilot_002_val_tokens",
            "path": WORKSPACE / "artifacts/stage7_integrated/navigation_head_pilot_002/validation/tokens.npy",
            "type": "file",
            "regenerator": "scripts/train_navigation_pilot.py cache generator",
            "precondition": "raw navigation data intact"
        },
        {
            "name": "navigation_head_pilot_003_tokens",
            "path": WORKSPACE / "artifacts/stage7_integrated/navigation_head_pilot_003/train/tokens.npy",
            "type": "file",
            "regenerator": "scripts/train_navigation_pilot.py cache generator",
            "precondition": "raw navigation data intact"
        },
        {
            "name": "navigation_head_pilot_003_val_tokens",
            "path": WORKSPACE / "artifacts/stage7_integrated/navigation_head_pilot_003/validation/tokens.npy",
            "type": "file",
            "regenerator": "scripts/train_navigation_pilot.py cache generator",
            "precondition": "raw navigation data intact"
        },
        {
            "name": "navigation_head_pilot_004_tokens",
            "path": WORKSPACE / "artifacts/stage7_integrated/navigation_head_pilot_004/train/tokens.npy",
            "type": "file",
            "regenerator": "scripts/train_navigation_pilot.py cache generator",
            "precondition": "raw navigation data intact"
        },
        {
            "name": "navigation_head_pilot_004_val_tokens",
            "path": WORKSPACE / "artifacts/stage7_integrated/navigation_head_pilot_004/validation/tokens.npy",
            "type": "file",
            "regenerator": "scripts/train_navigation_pilot.py cache generator",
            "precondition": "raw navigation data intact"
        },
        {
            "name": "resnet_navigation_pilot_005_tokens",
            "path": WORKSPACE / "artifacts/stage7_integrated/resnet_navigation_pilot_005/train/tokens.npy",
            "type": "file",
            "regenerator": "scripts/train_navigation_pilot.py cache generator",
            "precondition": "raw navigation data intact"
        },
        {
            "name": "resnet_navigation_pilot_005_val_tokens",
            "path": WORKSPACE / "artifacts/stage7_integrated/resnet_navigation_pilot_005/validation/tokens.npy",
            "type": "file",
            "regenerator": "scripts/train_navigation_pilot.py cache generator",
            "precondition": "raw navigation data intact"
        },
        {
            "name": "mixed_joint_010_tokens",
            "path": WORKSPACE / "artifacts/stage7_integrated/mixed_joint_010/train/tokens.npy",
            "type": "file",
            "regenerator": "scripts/train_navigation_pilot.py cache generator",
            "precondition": "raw 010 mixed data intact"
        },
        {
            "name": "mixed_joint_010_val_tokens",
            "path": WORKSPACE / "artifacts/stage7_integrated/mixed_joint_010/validation/tokens.npy",
            "type": "file",
            "regenerator": "scripts/train_navigation_pilot.py cache generator",
            "precondition": "raw 010 mixed data intact"
        },
        {
            "name": "mixed_resnet_010_tokens",
            "path": WORKSPACE / "artifacts/stage7_integrated/mixed_resnet_010/train/tokens.npy",
            "type": "file",
            "regenerator": "scripts/train_navigation_pilot.py cache generator",
            "precondition": "raw 010 mixed data intact"
        },
        {
            "name": "mixed_resnet_010_val_tokens",
            "path": WORKSPACE / "artifacts/stage7_integrated/mixed_resnet_010/validation/tokens.npy",
            "type": "file",
            "regenerator": "scripts/train_navigation_pilot.py cache generator",
            "precondition": "raw 010 mixed data intact"
        },
        {
            "name": "risk_features_014",
            "path": WORKSPACE / "artifacts/stage7_integrated/risk_features_014",
            "type": "directory",
            "regenerator": "scripts/cache_integrated_risk.py",
            "precondition": "raw risk branches and WAM intact"
        },
        {
            "name": "risk_features_015",
            "path": WORKSPACE / "artifacts/stage7_integrated/risk_features_015",
            "type": "directory",
            "regenerator": "scripts/cache_integrated_risk.py",
            "precondition": "raw risk branches and WAM intact"
        },
        {
            "name": "risk_features_hd03",
            "path": WORKSPACE / "artifacts/restart_stage7/risk_features_hd03",
            "type": "directory",
            "regenerator": "historical risk feature script",
            "precondition": "historical risk branches intact"
        }
    ]
    
    candidate_records = []
    seen_inodes = set()
    total_logical_bytes = 0
    total_dedup_bytes = 0
    
    for c in candidates:
        p = c["path"]
        if not p.exists():
            continue
            
        if p.is_dir():
            files_info = []
            dir_logical = 0
            dir_dedup = 0
            for root, _, filenames in os.walk(p):
                for f in filenames:
                    fp = Path(root) / f
                    try:
                        st = fp.stat()
                        dir_logical += st.st_size
                        if st.st_ino not in seen_inodes:
                            seen_inodes.add(st.st_ino)
                            dir_dedup += st.st_size
                        files_info.append({
                            "relpath": str(fp.relative_to(p)),
                            "size_bytes": st.st_size,
                            "inode": st.st_ino,
                            "nlink": st.st_nlink
                        })
                    except Exception:
                        pass
            candidate_records.append({
                "name": c["name"],
                "path": str(p.relative_to(WORKSPACE)),
                "type": "directory",
                "file_count": len(files_info),
                "logical_bytes": dir_logical,
                "logical_gb": round(dir_logical / (1024**3), 4),
                "regenerator": c["regenerator"],
                "precondition": c["precondition"],
                "dry_run_action": "PRESERVED_DRY_RUN_ONLY"
            })
            total_logical_bytes += dir_logical
            total_dedup_bytes += dir_dedup
        else:
            try:
                st = p.stat()
                file_bytes = st.st_size
                candidate_records.append({
                    "name": c["name"],
                    "path": str(p.relative_to(WORKSPACE)),
                    "type": "file",
                    "logical_bytes": file_bytes,
                    "logical_gb": round(file_bytes / (1024**3), 4),
                    "inode": st.st_ino,
                    "nlink": st.st_nlink,
                    "regenerator": c["regenerator"],
                    "precondition": c["precondition"],
                    "dry_run_action": "PRESERVED_DRY_RUN_ONLY"
                })
                total_logical_bytes += file_bytes
                if st.st_ino not in seen_inodes:
                    seen_inodes.add(st.st_ino)
                    total_dedup_bytes += file_bytes
            except Exception:
                pass
                
    protected_manifest = [
        "outputs/data/stage2_v2_100k/{train,val,test}/*.npz (Raw baseline dataset)",
        "outputs/data/stage2_v2_100k/manifest.json (Dataset metadata)",
        "outputs/data/stage2_v2_100k/normalization.json (Dataset normalization stats)",
        "artifacts/stage7_integrated/mixed_train_010/manifest.json (010 train metadata)",
        "artifacts/stage7_integrated/mixed_validation_010/manifest.json (010 val metadata)",
        "artifacts/stage7_integrated/navigation_train_009/ (Raw navigation training episodes)",
        "artifacts/stage7_integrated/traffic_train_009/ (Raw traffic training episodes)",
        "artifacts/stage7_integrated/navigation_validation_009/ (Raw navigation validation episodes)",
        "artifacts/stage7_integrated/traffic_validation_009/ (Raw traffic validation episodes)",
        "artifacts/stage7_integrated/control_wiring_mre_001/ (Control wiring MRE)",
        "artifacts/stage7_integrated/navigation_closed_validation_001/INVALID_CONTROL_WIRING.json (Failure evidence)",
        "artifacts/restart_stage6/runs_orinx/ (Stage 6 best.pt and last.pt checkpoints)",
        "artifacts/stage7_integrated/review_014_001/REVIEW.md (Stage 7 Run 014 review)",
        "artifacts/stage7_integrated/review_015_001/REVIEW.md (Stage 7 Run 015 review)",
        "docs/STAGE7_FAILURE_ANALYSIS.md (Retrospective analysis)",
        "docs/legacy/* (Historical specifications and goal versions)"
    ]
    
    return {
        "status": "dry_run_completed",
        "deletion_executed": False,
        "total_candidate_logical_bytes": total_logical_bytes,
        "total_candidate_logical_gb": round(total_logical_bytes / (1024**3), 4),
        "total_candidate_dedup_gb": round(total_dedup_bytes / (1024**3), 4),
        "candidate_items": candidate_records,
        "protected_items": protected_manifest
    }


def main():
    print("Executing Phase 0 audit...")
    sys_hw = audit_system_and_hardware()
    git_prov = audit_git_and_provenance()
    data_audit = audit_datasets()
    cleanup_preview = audit_cleanup_candidates()
    
    provenance = {
        "protocol_version": "BEV_WA_ALIGNED_V1",
        "date_user_timezone": "2026-10-22",
        "timezone": "America/Los_Angeles",
        "git_commit": git_prov["git_commit"],
        "is_dirty": git_prov["is_dirty"],
        "dirty_file_count": git_prov["dirty_file_count"],
        "hardware": sys_hw,
        "tracked_files": git_prov["file_hashes"]
    }
    
    resource_calibration_plan = {
        "hardware": {
            "gpu": sys_hw["gpu"]["device_name"],
            "vram_gb": sys_hw["gpu"]["vram_total_gb"],
            "backend": sys_hw["gpu"]["backend"]
        },
        "target_parameter_budget": {
            "ceiling_params": 100_000_000,
            "target_range": [50_000_000, 80_000_000],
            "encoder_allocated": 12_000_000,
            "joint_predictor_allocated": 55_000_000,
            "ema_target_encoder": "Not counted in deployment budget (training only, detached)"
        },
        "latency_target": {
            "frequency_hz": 10,
            "full_decision_p95_ms": 50.0,
            "included_components": [
                "BEV tensor CHW uint8 to float preprocessing",
                "Spatiotemporal encoder forward pass",
                "Joint World-Action Flow ODE integration (Euler/Midpoint 5-10 steps)",
                "Kinematic & safety bounds verification",
                "Controller waypoint tracking"
            ],
            "excluded_from_model_latency": [
                "MetaDrive environment step simulation",
                "Rerun logging and disk I/O",
                "Sensor capture"
            ]
        },
        "calibration_steps": [
            "1. Small-batch forward/backward smoke test on AMD RX 7900 XT (batch size 16, 32, 64) with bfloat16/float32.",
            "2. Peak VRAM measurement to ensure training VRAM stays well within 20GB without OOM.",
            "3. Single-item batch=1 inference latency profiling across 5-step and 10-step Euler/Midpoint solvers.",
            "4. Numerical gradient check for asymmetric backprop (L_world stop-gradient to action; L_action through world)."
        ]
    }
    
    with open(ARTIFACT_DIR / "provenance.json", "w") as f:
        json.dump(provenance, f, indent=2)
    with open(ARTIFACT_DIR / "data_audit.json", "w") as f:
        json.dump(data_audit, f, indent=2)
    with open(ARTIFACT_DIR / "cleanup_preview.json", "w") as f:
        json.dump(cleanup_preview, f, indent=2)
    with open(ARTIFACT_DIR / "resource_calibration_plan.json", "w") as f:
        json.dump(resource_calibration_plan, f, indent=2)
        
    print(f"Phase 0 artifacts written to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
