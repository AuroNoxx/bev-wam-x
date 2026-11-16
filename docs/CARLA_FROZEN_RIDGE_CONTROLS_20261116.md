# Frozen ridge controls 2026-11-16
Completed23.11s, unchanged backbone checkpoint hash; no neural training, closed loop, official test, guards or defaults changed. Artifacts carla-frozen-ridge-controls.

Predeclared alpha state0.01, context1.0 (100x previous). Same train1570 unique windows, train-only standardization, dev fixed. State_interaction adds only product(current speed/30, near-goal lateral/50) to existing3state features; four-dimensional ridge. No new observation, future or script timer. Context still384 pooled dimensions; no dev hyperparameter selection. Both controls reported, not cherry-picked best gain.

Early corrective train155; dev82 (41 each sign). Stateinteraction trainMAE0.01001m,dev0.00939m, directions97.4% on significant targets; positive/negativeMAE0.00938/0.00941m. Predicted absolute amplitude0.03289 vslabel0.03510m. OriginalstateMAE0.03006; FM4candidate-mean0.03592; zero0.03510. Recovery static188: interactionMAE0.001462 vsstate0.005448 vszero0.000552 (remaining false drift). Near-center moving264:0.000954 vsstate0.002068. Original214/startstop314 entiredev interactionMAE0.000419/0.000304.

Strongercontext regularization devearlyMAE0.03040, direction93.6%, vs previous0.04859/50%. Thus previous poorcontext readout partly attributable to high-dimensional probe overfit. Current results do not establish encoderinformationloss. Still no amplitude comparable to stateinteraction and no fullpolicy proof.

Causal current state supplies useful short-horizon lateral conditional interaction in this limited domain. CurrentFM fails to exploit it reliably; data are not simply devoid of signal. This scalar0.6s readout omits longitudinal speed/intent, whole5s dynamics, obstacle interaction and safety. It must not be sent to controller as a driving policy. Reuseddev and one frozenmodel/limitedroads remain. No automatic next training.
