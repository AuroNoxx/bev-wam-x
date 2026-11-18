# Frozen context motion probe

Completed scripts/probe_reactive_frozen_motion.py in42.45s. Artifacts/reactive-frozen-motion-probe/report.json. Four2070/2071 base/history checkpoints frozen eval/no-grad, checkpointSHA256 unchanged. No neural training, no CARLA execution, no test access.

## Contract
Actual actor4frame OCC plus permitted state only. Same last-issued cruise/permission, current navigation/speed, and acceleration only for historyarm. Audit data actor adapter excludes ID/leadpose/gap/velocity/teacherdecision; rawcoords only undergo originalvoxelization. Forbidden future-label/teacher-extra-field poison does not change context.

Prediction target is past0.6s lead average forwardmotion, not futureaction. GT used only readout supervision/evaluation. Train514 onroad388;dev514 andseparate audit414onroad713. Auditsharesdevseed4101, notindependentscenario. Train/devcruise8/8.5androadconfounded.

Freeze pooled context: mean160spatialtokens per current/+1/+2predicted block,384features; futureblocks are modelpredictions, notfuturetruth. Fixedalpha1 meanMSE ridge,trainstandardization,unpenalizedintercept; no dev tuning. Comparestate7,context,context+state,zero-motion andno-ID geometricOCC. Poolingcan discard spatialinformation; failureofthisreadout doesnotprove originalencodercontainsnone.

## Common geometry-valid audit412frames, MAE m/s
| checkpoint | context | state | context+state | OCC geometry |
|---|---:|---:|---:|---:|
|2070base|4.004|1.976|3.799|0.270|
|2070history|3.077|1.975|3.033|0.270|
|2071base|2.924|1.976|2.317|0.270|
|2071history|1.061|1.975|1.069|0.270|

ContexttrainMAE0.386..0.461;contextdev0.985..4.044. Trainreadoutfit substantiallybetter thanheldresults.2071history preserves usefulmotion readout relative tostate-only, but notstableacrossseeds andstillworse thangeometricinputreadout. Staticabsoluteerrorcontext0.923..3.489m/s versusgeometry0.207: no robuststationarybehavior claimed. Geometryitself weakonverylowmotions andhasunknowns; tableusescommonvalidsubset so coverageisnotignored.

## Interpretation
Data contains accessibleordinarymotion. Frozenpooledrepresentation/readout is not reliablytransportable acrosscurrenttrain/devepisodes. Cannotattribute failureuniquelytoencoder, pool, regularization, insufficienttraining orroute shift; no evidencejustifiesanotherblindFlowloss change. Nextifrequested: a pre-fixedspatiallypreserving frozenreadout control, orbroader independentepisode validation, notautomaticpolicytraining. No claim ofdrivingacceptance orcontrol utility.
