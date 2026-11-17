# Native OCC-Qwen v2 final audit (2026-11-17)

## Completion and integrity

Completed 20 epochs, 13240 optimizer steps in 4267.58 seconds (71.13 minutes). Status callback displays epoch21 after completion; checkpoint and 20 unique validation records confirm actual20 epochs. Original trainer processes exited normally. CPU audit passed finite model/optimizer tensors, scheduler/optimizer step consistency, EMA structure, RNG and exact loader resume states. best is epoch17/step11254; latest is epoch20/step13240. They are distinct and both retained.

## Matched validation

Same1124 ordered Town13 validation windows; batch8; single candidate;8Euler steps; dt0.2s; independent generator1907+batch_index. No test split or CARLA closed-loop evaluation. v1 original production-stream ADE4.575159 is not used as this comparison baseline.

| Checkpoint | Epoch | ADE m | Speed MAE m/s | Vector jerk mean | P95 | Max | Points >10 | Trajectories any >10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v1 best | 20 | 4.462059 | 2.161276 | 10.505564 | 23.202366 | 163.977692 | 43.312% | 99.911% |
| v2 best | 17 | 3.171468 | 1.622556 | 8.615677 | 24.959929 | 280.438904 | 26.922% | 92.794% |
| v2 latest | 20 | 3.831426 | 1.886403 | 9.827580 | 29.138920 | 212.410675 | 31.189% | 97.242% |

v2 best ADE improves 28.92% versus matched v1, and mean vector jerk improves 17.99%. However jerk P95 and maximum are WORSE. Average below10 does not imply threshold compliance:92.794% of best trajectories still exceed10 somewhere. Best selection is by validation ADE, not safety. Epoch18 ADE rebounded4.460676 and latest3.831426: the run remains validation-unstable; no causal diagnosis is established.

## Endpoint errors and World

- v1 best: endpoint errors1/2/3/5s = 1.137908/3.065303/5.164943/9.646008m.
  World prediction1s/2s = {'1s': 0.2586453159229933, '2s': 0.2808919878646636}; own persistence = {'1s': 0.5074931467991319, '2s': 0.7623806937356972}.
- v2 best: endpoint errors1/2/3/5s = 0.809381/2.105829/3.572049/7.152280m.
  World prediction1s/2s = {'1s': 0.06511271019997067, '2s': 0.13505788869933982}; own persistence = {'1s': 0.11474188510275045, '2s': 0.1788991660564316}.
- v2 latest: endpoint errors1/2/3/5s = 0.956698/2.477580/4.283610/8.688366m.
  World prediction1s/2s = {'1s': 0.06344518559740708, '2s': 0.11922445405448458}; own persistence = {'1s': 0.11594012451150075, '2s': 0.17900916648776188}.

World scores use the SAME checkpoint EMA encoder and current ego frame, occupied-support mask on its own lattice. Each beats its own persistence. Raw latent MSE is NOT comparable across encoders/lattices; neither absolute magnitude nor beating persistence proves World improves Action. No future OCC decoder or OccWAM stage2 was added.

Position-derived and velocity-vector jerk agree within floating-point precision. Scalar-speed jerk is separately reported in evaluation JSON; it omits directional changes and cannot replace vector jerk.

## Evidence and preservation

- artifacts/occ-qwen-native-v2-final-audit/checkpoint-audit.json
- artifacts/occ-qwen-native-v2-final-audit/best-evaluation.json
- artifacts/occ-qwen-native-v2-final-audit/latest-evaluation.json
- scripts/audit_occ_qwen_native_v2_final_checkpoints.py
- scripts/evaluate_occ_qwen_matched.py

Both best/latest full checkpoints (~1.320GB each), all source data/native aligned cache, source snapshots, logs and reports retained. Formal run contains no surplus intermediate checkpoint files. The unique GPU resume-test split checkpoint remains evidence, with no verified equivalent replacement established. No v2 files were deleted: additional released bytes0. Previous v1 lossless archive cleanup released5,637,786,797bytes and remains restorable. No further model version or training automatically started.
