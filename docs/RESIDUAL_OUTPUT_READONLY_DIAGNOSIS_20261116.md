# Read-only residual output diagnosis

Input: already saved42diagnosticwindows x4candidates perarm, no rerun or updates. Script scripts/analyze_residual_outputs_readonly.py. Results artifacts/residual-output-readonly-audit/report.json. No test.

## Backward endpoint is not a backward prior
All42 current speeds positive:0.182–12.693m/s. All42label endpoints positive:8.124–63.875m. Mean current-v0 5s prior+45.6445m,meanlabelend+45.5221m. Seed1981 residual predicts end correction−57.5677m(normalized by final xscale31.872 gives−1.8062),overwhelming thepositiveprior and yielding−11.9232m. Even v0>=5m/s groupend−9.6096m versuslabel+46.8743m. This cannot be explained solely by near-stop cases or prior sign.

## Heading and temporal-position failures are distinct
1981 residual predictedcos ranges−1.0..−0.7988,mean−0.97975: EVERY sampledpoint faces backward. Mean first/lastcos−.97953/−.97912. F.normalize cannot flip sign,so this is an output failure rather than sign inversion by normalization. These are immature128-step modelpredictions; do not label a general training attractor without additional evidence.

Other3arms alwayscos>0 but backward positionalsegments remain:1980original47.43%,1980residual28.81%,1981original51.21%. Thus yaw flips alone do not explain all backward differences: independent waypoint positions and independent speed/heading heads are inconsistent.

## Position-derived versus speed-head dynamics
Mean pertrajectory peak jerk(m/s3), position-derived / speed-head:
-1980 original9996.73 /666.20
-1980 residual3684.73 /847.50
-1981 original23939.92 /428.45
-1981 residual7451.55 /785.11
Position x second-difference RMS(m):16.34→6.13 and40.71→12.80. Residual conditioning reduces position oscillation but it remains enormous; speed-headjerk worsens. Earlier summary jerk referred to speed-headjerk and must not be mistaken for vehicleexecutedjerk. Position-derived derivative assumes piecewise intervals. Neither is actual control or measured vehicle dynamics.

## Code-level interpretation and next falsifiable step
The unitheading penalty enforces length, not correct hemisphere: both[sin0,cos+1] and[sin0,cos−1] satisfy it. Kinematic consistency likewise can reward mutuallybackward positions and velocity rather than route-correct motion. FMtargets still distinguish correctheading; current output demonstrates that128updates at theend ofwarmup have not learned it reliably. This is a mechanism hypothesis,not gradient attribution proof.

Priority next frozen check: for the failedseed compare raw sine/cosine trajectory through flowtime and decompose alreadyweighted FM/position/kinematic/unitheading gradients around the observed heading branch; avoid prescribing extraheadingloss before evidence. Also explicitly separate absolutepositionoffset,local second-difference jitter and predictedspeedhead errors in the samewindows. A bounded longer same-budget comparison could distinguish immatureoptimization from representational defect,but is not started or authorized here. Do not adopt residualdefault,hardclipbackwardmotion or proclaim integrateddecoder a solution by construction. Prior integrateddecoder results remain relevant. No newtraining began.
