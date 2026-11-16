# Two failure hypotheses and one bounded next test

Date: 2026-11-16. Analysis and frozen tests only; no new training, collection, or closed-loop run.

Evidence: artifacts/matched6-ambiguity-shape-v2/report.json and identical_actor_loss_floor.json. V2 includes exact history masks and ages in actor hashing. The earlier v1 actor hash omitted those arrays and is superseded; tensor-by-tensor equality of every actual actor field was independently verified in the loss-floor audit. Code: analyze_matched6_ambiguity_shape.py; audit_identical_actor_loss_floor.py. Existing checkpoint hashes unchanged.

## Hypothesis A: teacher future is not uniquely observable
137 stationary train windows meet current speed <.05 m/s and absolute past-.6s acceleration <.05 m/s2. Five groups (11,27,17,38,36 windows) have identical complete actor tensors: four historical OCC frames, state, history masks/ages and derived geometry-valid mask. Yet their future labels differ. The 38-window group has 5s speed labels1.487..6.505m/s; the 36-window group2.037..6.765. No deterministic model receiving those identical inputs can recover each individual future. Lead schedule must remain hidden; do not add elapsed phase/countdown as a shortcut.

This is stronger evidence than similar-looking images or instantaneous GT lead speed. It does NOT mean the immediate reaction task is unlearnable, or every failure is irreducible. Physical occupancy visibility is not reliable early-motion detection.

## Hypothesis B: conditional future-time shape is weakly learned
After subtracting each predicted trajectory's own mean longitudinal velocity, the leading temporal mode explains99.74..99.87% of energy across all held inputs, both seeds at512and1024. The mean start/decel predicted temporal-shape cosine is+.9985..+.9990; label shape cosine is−.9923. Predictions largely change their velocity level, while reusing the same slight time-dependent curve even for opposite required acceleration trends.

The sign here uses longitudinal vx, NOT speed magnitude. Thus apparent speed increases from a growing backward velocity cannot masquerade as correct acceleration. At1024, start mean vx change between.2s and5s is−.124/−.094m/s, decel−.130/−.103, whereas labels start+5.202/decel−3.391.

Earlier time-permutation and zero-position tests showed the physical-time axis is present. Every time embedding receives gradients and changes. Existing action/position gradients were aligned, not cancelling; world gradient was smaller. Therefore a time-index coding error is not supported, and neither deleting JEPA nor assuming infinite extra steps is justified.

## Both hypotheses coexist, but multimodality is not enough
For each exact-input group, normalized velocity action MSE decomposes exactly:
MSE(prediction, labels) = variance(labels around empirical group mean) + MSE(prediction, group mean).
This algebra was verified numerically using actual frozen1024predictions, with no fitting. Across the five groups and two seeds,61.1%..98.4% of current action MSE is the second, excess term. Empirical conditional-mean curves themselves rise from approximately0 at.2s to4.57..6.49m/s at5s; the model has not learned even that time-varying compromise.

This floor applies to action MSE on these finite duplicate groups only. It is not the optimizer of the combined position/world loss, a population Bayes-error estimate, or a safe driving target. A mean future can still be an inappropriate realized driving action.

## One proposed falsifiable next test (NOT EXECUTED)
A decoder-only learnability unit test, not a new driving experiment:
- Keep actual encoder/context frozen from each existing checkpoint, and retain the original Transformer decoder/time-position mechanism and25x2 output representation.
- Use five exact-input group contexts with their empirical mean velocity targets, plus five preselected distinct train deceleration contexts/labels. These are a fixed tiny supervised function, not a new dev set. No future enters context.
- Optimize only the original decoder-side parameters against action MSE for one preset short budget (e.g.256updates, full fixed batch10, two existing seed initializations). The removal of joint auxiliary objectives is intentional isolation, NOT a fair architecture/quality ablation. Preserve all main checkpoints.
- Predeclare reporting: initial/final per-time physical errors, start vs decel curve direction, fit to zero/constant baselines and numerical stability. Fit alone is not driving success; no held generalization claim or CARLA follows.
- If original decoder can learn these opposite curves, its basic expressive/optimization path works in isolation; investigate shared end-to-end condition transfer/objective coupling rather than replacing the whole architecture or collecting more of the same data. If it cannot under the fixed budget, record the failure; this still does not prove mathematical incapacity, but supports focusing on explicit physical-time conditioning before another full training run.

Do not combine this diagnostic with new data, a new noise distribution, different losses in the main training system, or guard changes. No action beyond this analysis has been started.
