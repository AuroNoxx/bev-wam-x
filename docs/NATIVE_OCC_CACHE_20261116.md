# Native aligned cache branch (2026-11-16)

Independent files only; active trainer and configuration untouched.

Interface: NativeCachedDataset(index_path, 'train'|'validation', actor_only=False).
Returns exactly history/state/trajectory/future as NativeDataset. Native grids
remain 4x20x128x128 and 2x20x128x128, 0.4 m, +-25.6 m XY. No GPU use.

Cache index binds manifest, b2d_native_occ.py, cache implementation and every
original OCC, annotation and scalar-label file SHA256. Dataset construction
rechecks source hashes, item reads reject source stat changes and check object
SHA256. Writes are temporary/fsync/checksum/atomic-replace; prior objects and
indexes remain intact. This cache does not certify upstream source causality.

Pilot: train/validation each 3 samples, every returned tensor torch.equal.
CPU reads approximately 3.7-4.0 ms/sample vs raw 140 ms/sample, 35-39x; tiny hot
sample benchmark, not full training speedup. Dataset source validation ~28 ms.
Report: artifacts/native-occ-cache-pilot/report.json.

Process-restart exact StatefulDataLoader restoration verified separately at
workers=0 and workers=2. Cross-worker restoration is NOT supported: 0->2 lacks
_snapshot; 2->0 lacks _num_yielded. Do not discard state to change workers.
Legacy NativeDataset workers0 saved state -> NativeCachedDataset workers0 in a
fresh Python process also matches all remaining batch tensors on the three-item
subset. Manifest/order/batch/shuffle/seed are checked and recorded in
legacy-dataset-resume.json. This is not a full Lightning resume validation.
Before production migration verify full split order, checkpoint manifest and
settings, and keep batch size/shuffle/seed/workers unchanged.

Full build authorized by coordinator; background single CPU process, one torch
thread, BelowNormal priority; no changes to active training. Logs:
artifacts/native-occ-cache-pilot/full-build.stdout.log and stderr.log.
Final index path is printed only after successful atomic publication. Do not
change native_occ_cached_data.py or b2d_native_occ.py during/after building:
code changes invalidate this cache by design. Full validation must use final
index, exact full split membership/order, and sample raw equality checks.

## Lightning migration and epoch RNG correction

CPU-only tiny-backbone Lightning integration now covers 4 steps crossing an
actual 3-sample epoch, saving at step 2, resuming old NativeTask/raw data into
OptimizedNativeTask (performance flags false)/cached data. A pre-existing
TorchData shared-generator restoration defect was exposed also in raw->raw:
current-epoch order matched but next epoch differed. Its sampler snapshot is
pre-permutation RNG; restoration omitted the later DataLoader base_seed draw.

Independent native_occ_exact_loader.ExactResumeDataLoader saves actual shared
RNG state. Its explicitly enabled legacy reader supports ONLY validated workers0,
full no-replacement random sampler sharing the explicit loader generator; strict
legacy schema/counters are checked, then old current RNG is reconstructed by the
original randperm followed by base_seed draw. It retains all sampler position and
dataset state. Caller still checks settings, manifest and full dataset order.
Do not use it for validation sequential loaders or workers2.

With this loader, model/optimizer/scheduler/EMA/RNG and batch sequence are all
bitwise equal, max parameter difference 0. Both legacy and new-format restoration
pass. Evidence: artifacts/native-occ-cache-lightning-test-fixed/result.json and
artifacts/native-occ-cache-lightning-test-fixed-newstate/result.json.
Production pretrained GPU migration is not tested by these CPU tiny-model tests.

Integer-only production-scale regression (5296 items,batch8,seed71,workers0)
checks legacy checkpoints at batches2/650/662, before StopIteration and after
_iterator_finished=True, then full next two epochs. Every index matches. New
Exact->Exact twice-resumed continuation also matches. Evidence:
artifacts/native-occ-cache-pilot/integer-loader-regression.json.
