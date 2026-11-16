# Route-isolated regression / Flow offline diagnostic

Authorization: parent relayed user continuing bounded comparison, 2026-11-16. No default change or long training.

Scanned all 5296 training windows. Town+Route IDs (including duplicate weather recordings) split with fixed seed2050 per scenario: 3683 training,785 tuning,828 held development. No official validation or test used. Different route IDs can share physical road geometry; all data has previously been inspected in development, not a new final holdout. Within each diagnostic route, select non-overlapping complete history+future support intervals (nominal frame chronology). Tuning16, held17, train diagnostic20 windows. New models, not old checkpoints.

Two paired seeds2050/2051, counterbalanced FM/direct-regression arm order. Equal-scene shuffled cycles:12288 samples/arm, all3683 training files covered, repeats explicit. Same data sequence and initialization per pair, fixed noise/time RNG, batch12,1024 updates, AdamW1e-4,64step warmup then cosine to1e-5 at1024,clip1,BF16. Same current-support JEPA auxiliary losses; primary action objectives differ and totals cannot compare as quality. Width128/layers2/heads4,966939 trainable parameters; velocity2 integral decoding unchanged. Regression deterministic, repeated candidates not independent. FM16Euler pure-noise generation. Validation8Euler monitors diagnostic label position only; final evaluations16Euler/latest1024. Best is not best-driving.

Bound:8minutes/arm,35minutes wall pre-arm limit; nonfinite gradient/loss stops. Resource prior about1GiB allocated,12-18min overall estimate, not guaranteed. Latest/best saved through Lightning+VerifiedCheckpointIO+StatefulDataLoader, TensorBoard; old weights untouched. First invocation import-name typo failed before training, corrected.

Report position fit only as diagnostic; speed/position consistency, t0/full acceleration and jerk, progress, stop fraction, candidate diversity, per-scene summaries. Collision/TTC/road compliance/closed-loop safety unknown. Current-source OCC past-only generation unverified; no online safety claim and unknown cells are not free-space. Directory scenario labels do not certify actual events. Small model/data study does not establish155M model quality.

## Completed results
All four arms completed1024, no automatic extension. Total820.19s(13.67min),197.16–205.98s per arm; peak allocated0.94004GiB. Both seeds covered3683/3683 training files via12288 samples. First3paired action RNG traces match. Actual full route split excludes window/support crossing within route, but physical geometry across distinct route IDs is unverified.

Held17 support-disjoint windows, raw candidate mean, FM -> regression:
- seed2050 position diagnostic5.321->3.974m; vector velocityRMSE2.616->2.017m/s; mean per-candidate peak vector acceleration24.112->9.322m/s2; jerk204.115->47.292m/s3; forward31.146->29.256m.
- seed2051 position6.779->5.065; velocity2.991->2.315; acceleration24.766->4.965; jerk209.914->25.832; forward27.515->26.271m.
- Held label forward31.778m. Regression improves position diagnostic in all six directory classes in both seeds, but reduces forward progress, remains inaccurate and nonphysical; no task success conclusion.
- FM candidate spread4.555/4.991m; deterministic regression0 by definition, not evidence of better coverage or multimodal behavior.
- World predicted/persistence within each own EMA:2050FM .442/.298,regression .458/.296;2051FM .244/.140,regression .262/.134. All worse than persistence. Each arm's EMA defines a different space; these numbers are not a common fixed cross-arm quality ranking. Source/mask limits unchanged.

Interpretation: broad training support now corroborates additional generative/noise burden at this small model and budget. Does not establish sufficient perception or deployable regression. Current-sourceOCC causality unverified and no collision/TTC/closed-loop/safety evidence. Keep regression as diagnostic baseline, do not change default. A next decision should separate generative initialization/conditioning and trajectory representation from shared world failure, rather than assume more steps or155M will fix them.
