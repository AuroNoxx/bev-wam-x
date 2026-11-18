# CARLA bounded rule and event calibration, 2026-11-16

No learned model or LLM was used. Three owned-actor, low-speed runs in Town10HD_Opt, fixed 0.05 s, same configured straight 15 m static-obstacle scenario. Privileged simulator geometry supplies approximate aligned bumper gaps; this is not OCC perception validation. Fingerprint identifies configured setup and paired-controller protocol, not guaranteed bitwise dynamics, and not a replay of identical measured initial states. Each actor was respawned and settled for 20 ticks.

- Positive contact calibration: 124 recorded frames, 124 matching IMU frames; one collision callback against the intended parked vehicle, impulse norm 3115.93. Stopped on callback. This demonstrates this stimulus is observable, not complete collision coverage.
- Gap-braking rule, target 3 m/s: 160/160 frames (8 s), zero raw callbacks, progress 6.714 m, min TTC 1.765 s, mean absolute speed acceleration 0.601 m/s², mean absolute speed jerk 2.903 m/s³.
- Crawl-braking rule, target 1.5 m/s: 160/160 frames, zero raw callbacks, progress 6.474 m, min TTC 3.176 s, acceleration 0.311 m/s², jerk 1.910 m/s³.

No aggregate winner. Zero callbacks is not a certified collision-free metric. Collision coverage, offroad/redlight/intervention rates, route completion, world quality, planner latency and LLM assessment remain unknown. Constant rule braking is not a safety takeover. Scalar speed derivatives do not fully describe comfort. Calibration contact run has a shorter duration, so its progress cannot be ranked against paired runs.

22 contract tests pass. All owned vehicles/sensors removed; original world settings restored and readback found zero vehicles/sensors. Own server was closed after checks. No weights, test data, or training touched.

Artifacts: artifacts/carla-rule-event-calibration-20261116/report.json plus three full episode records. Script: scripts/carla_rule_event_calibration.py.

## Astra high integration boundary
User requests Astra high reasoning. Internal assistant orchestration can assess selected episodes asynchronously through an Astra high worker. It is not a Python-accessible provider API and no such integration is claimed. No credentials read and no external inference performed. Next offline judge input should include raw episode samples, observed events, masks, scenario metadata and explicit unavailable fields; output structured evidence-grounded critique, uncertainty, suggested controlled scenario changes, never a safety override or unverified reward. Automatic training-loop integration still needs a supported endpoint/provider/model mapping and authorized credentials/budget.
