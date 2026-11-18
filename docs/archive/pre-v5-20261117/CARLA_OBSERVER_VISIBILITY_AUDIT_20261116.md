# Observer visibility read-only audit and next contract

## Existing hold_v3 evidence
Artifact: artifacts/reactive-observer-visibility-audit/report.json; script scripts/audit_reactive_observer_visibility.py.1200rawframes read,514validanchors per split summarized. No new training or simulator tick.

Global unknown99.25% does not imply frontcar invisibility. Every anchor has vehicle-semantic rays in current ego forward corridor x3..60m, abs(y)<1.75m after existing self-filter. Train12..119 rays, median25,6..24uniquevoxels; dev14..120,median24,6..24voxels. Stop/hold also has rays. Same-frame nearest vehicle surface x vs recorded bbox bumper gap has Pearson>.999996, with near-constant2.30..2.33m origin-to-bumper offset. This supports a lead-related distance signal, not proof of identity or model understanding.

Actual instance lead visibility remains unknown: savedNPZ has xyz/tag but no object_idx, no saved lead ID/pose/bbox. Existing sparse observation cannot be blamed as wholly uninformative based on volume-wide unknown.

## Next independent instance audit, no actor augmentation
1. Match sensor/snapshot frame and timestamp, object_idx to same-frame actual lead actor ID; record association validity, do not silently accept missing or ambiguous IDs.
2. Count actuallead rawpoints, semantic labels, after self-geometry filter, after model range/cell quantization, and final class4 voxel support. Distinguish missingpacket/occlusion/outsideFOV/grid/unknown/unverifiedidentity. No hit does not imply free or no lead.
3. Stratify distance, motion and event; report episodes/event counts as well as correlatedframe counts. Training anddev roads separate; tinyaddedepisode is audit not representativeevaluation.
4. For past0.6s only, transform each historical lead point set to the same current ego frame used byactor. Compare lead-associated audit voxel support/centroid/Jaccard over history against true pastleadmotion. ID-based correspondence is oracleupperbound, not a student tracker.
5. Particular sensitivity: lead restart .258m/s over.6s=.155m, below .8m voxel size. This alone does not prove invisibility because boundary points can change; directly report tensor/support differences and missing observations.
6. Keep exactteacher observationf andpostrawf+1 alignment. Compare like-frame velocities/distances, not accidentalfuture. Any finite difference must use timestampeddelays and sameframecoordinates.
7. Separately report ID-free actor features vs oracle audit. Never copy ID/leadpose/bbox/gap/velocity into actor to improve results silently. Policy forecast cannot be scored as actualexecuted safety.

## Minimum next decision
Only after audit distinguish (a) actual lead hidden or quantization erases motion, versus (b) useful observable distance/motion exists but encoding/generation fails. Do not select another sampler/loss change without this evidence. No training/closedloop authorized by this audit plan itself.
