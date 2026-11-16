# Authorized continuation 155M, 256 -> 2048

User authorized continuation. Source first256/best.ckpt has model, optimizer moments, scheduler, RNG and yielded-batch256 loader cursor. Original source files untouched. New output artifacts/masked-teacher-155m-continue2048.

Explicit schedule fork, NOT exact old-schedule resume: reset LR from terminal1e-5 to5e-5; cosine decay to5e-6 over1792 new updates, retaining AdamW moments/model/RNG/loader/loop state. Generated continuation-start.ckpt records new task settings and schedule, verified before fitting. Lightning reports all states restored. Fifteen768-wide12-head decoder layers,155221278 nonEMA parameters; b8,noaccum,BF16,4 masked/time-aware historyframes. Same5296train windows and64fixed validation, no test. Other loss/EMA settings unchanged.

Max global step2048, wall budget25min. Latest every128updates, validation every256, best validation_loss is only an offline fitting checkpoint selection. JEPA prediction vs persistence logged during validation. Scalar motion inconsistency diagnostic logged; collision/takeover/closed-loop/Astra-RL remain unknown/false. No learned CARLA runs or external LLM calls. Source OCC causal certification remains unresolved.

Observed288:loss1.0100,LR4.99646e-5,16.71s since train start including~9s restore overhead. Heat segment31updates/7.615s≈0.246s/update. Peak allocated4.221GiB/reserved4.615GiB. Estimated10–15min including repeated evaluation/durable snapshots; not a guarantee. No restart on failure and no extension past2048.

## Completed and read back
2048/2048 normal completion. Added1792updates; last-update timer531.29s (8.85min) including earlier checkpoint/evaluation overhead, final validation/save additional. Peak allocated4.236GiB/reserved4.615GiB. Final validation total0.215138,FM0.069857,kinematic sampling residual2.04052m/s,ADE5.54790m diagnostic only. JEPA latent0.228324 vs persistence0.165692 remains worse, so world predictor not accepted.

best/latest bothstep2048, both store current-epoch loader yielded62 (2048 mod662), not zero-reset. Strict model/config/RNG load succeeds. Replaying the actual continuation (restore source256 state, then1792 updates) produces identical next8 sample indices [371,4402,1248,1244,5061,2510,2877,915] to both restored snapshots. A different experiment replaying2048 from seed without resume differs due sampler lifecycle, and is explicitly not claimed equivalent. No extra optimization steps run for this check.

Fixed16-window postfit candidates (same4 candidates/16Eulersteps as256 probe) are finite and actor label isolation passes. Kinematic residual1.687–2.038m/s; mean |scalar acceleration|2.555–3.518m/s², maximum11.663–18.917m/s², mean |scalar jerk|21.682–30.109m/s³. Improved relative to initial256 but spikes remain; predicted, not executed, no comfort/safety pass. Latent and semantic prediction both still worse than same-support persistence on all4 batches. Full candidates for two inputs are retained in postfit-evidence/report.json for Astra. No collision/offroad/takeover measurements.

Process exited and budget not extended. All original first256 and older weights retained. Test not read; source-OCC causality unresolved; no automatic LLM RL or learned-CARLA driving.
