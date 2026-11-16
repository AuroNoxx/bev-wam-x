# Frozen deterministic readout diagnostic 2026-11-16
Artifacts carla-frozen-ridge/{design,report}.json. Completed22.94s, checkpoint SHA unchanged. No backbone updates, test set, closed loop, default or guard changes.

Fit1570 unique actualtrain windows only (same mixture exposure); fixed alpha0.01 for mean squared y06 plus coefficient L2 after train-only standardization/intercept. State3features: current speed/near-x/near-y. Context384features: spatial mean of160tokens per current,+1,+2 block. Standardization floor1e-5; hyperparameters not selected using dev. This pooled linear probe can lose spatial information and can overfit; failure is not proof that full representation lacks information.

Independent dev road713 recovery early-corrective82 windows (current moving, index<=120, |near-y|>.01): target meanabs0.03510m. MAE zero0.03510,state0.03006,context0.04859,FM4candidate mean0.03592. On nontrivial targets state sign97.4%, context50%,FM53.8%. State amplitude only~0.005m; it underpredicts and produces erroneous static output~0.0059m. Context train corrective MAE0.01625 vsdev0.04859 reveals severe generalization/readout problem. Original/startstop dev context errors0.0424/0.0465m vsFM0.00999/0.01025m. Repeateddev, one checkpoint, sparse oracle OCC restrictions remain.

Zero baseline often wins global MAE due mostly straight/stopped samples; that does NOT mean zero driving is good. FM4mean is a statistical diagnostic, not a trajectory the controller executed. No safety/efficiency claims.

Next proposed only: disambiguate low-dimensional state speed interaction vs pooled-context high-dimensional readout overfit using pre-fixed lower-dimensional/regularization controls. Do not automatically retrain FM or sweep navigation gain. Data intent redesign remains separate; no future countdown is permitted in actor. Current go/hold may be valid exogenous current instruction but cannot uniquely determine all subsequent command changes.
