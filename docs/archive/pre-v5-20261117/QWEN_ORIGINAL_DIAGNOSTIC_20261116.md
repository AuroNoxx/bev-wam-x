# Original Qwen inference diagnostic (2026-11-16)

No training, weight edits, CARLA restart or vehicle controls. Frozen original Qwen3-VL-2B-Instruct BF16, local model, MATH SDPA. Six saved v4 CARLA cases: follow/brake/lanechange repeat0 control indices0 and30. Exact history_frame_ids map to four saved RGB PNGs, each448x252; 0.2s gaps. state20 decoded to physical-unit text. Original frame hashes, prompts, outputs and prior driving-head outputs saved in artifacts/qwen-original-vlm-diagnostic/inputs.json and outputs.json. The short prompt actual rendered strings are in short/outputs.json (short/inputs.json retains original prompt metadata).

This is added chat-template/text input with language output, not the same tensor interface as the trained driving head (four vision markers without instructions and pooled hidden features). No direct equivalence or causal isolation of head/backbone failure.

First arm: greedy, max300 tokens, default repetition penalty1.0. Four follow/brake examples repeated until300 tokens. Lanechange outputs more coherent but inferred vehicles/motion without adequate evidence. Peak allocation4.242GiB.
Second diagnostic arm: short four-line prompt, greedy, repetition_penalty1.1,max160. All six ended without truncation,1.74–2.99 seconds per sample. Both prompt and decode changed, so not an isolated ablation.

Verified observations:
- Follow0: correctly recognizes black sedan ahead and road curving right in short output. Motion statement is uncertain/inaccurate; cannot quantify driving capability.
- Brake30: measured ego2.18788556m/s, gap8.8737341m, TTC4.0558493s; lead speed derived from logged gap/TTC and ego speed is exactly0m/s. Scenario code sets lead target velocity0 and full brake after tick80. Original Qwen says 'moving forward at a moderate speed' in long arm. Short arm says 'The vehicle is moving forward along a straight path' and 'Since the current speed is 2.19 m/s ... continue to accelerate slightly', followed by maintaining course/speed. This appears to confuse supplied ego telemetry with lead motion. It does not reliably identify stopped lead and appropriate slowing.
- Lanechange0: short arm correctly says no visible vehicles and road curves right, but refers to left lane-change command as a 'planned left turn' and recommends slight right steering/maintaining speed; long arm recommends left steering. Navigation semantics/prompt response is unstable.
- Lanechange30: describes a white van far right, not independently verified from actor ground truth; do not present this as accurate perception.

Conclusion: pretrained 2B at these low-resolution four-frame conditions produces recognizable scene descriptions but unreliable motion/navigation reasoning and recommendations. Cannot say original Qwen works and only Flow is broken, nor that Qwen inherently cannot drive. Small diagnostic, prompts/decoding/resolution are confounds. No closed-loop test of text policy.

Reproduce: .venv-train/Scripts/python.exe scripts/diagnose_original_qwen_vlm.py; then scripts/diagnose_original_qwen_vlm_short.py. Both exited successfully; no continuing inference/training.
## History-length diagnostic
No project train/cache/worker processes existed before this diagnostic; no new training started.
Current inputs DO include ego physics: speed, forward/left acceleration, yaw rate, near/far navigation coordinates and commands. No OCC, lead 3D kinematics, calibrated object distances, dense map geometry or validated physics world model are provided.

Twelve original-model inference cases: three scenarios x two repeats x 0.6/3.0-second four-frame histories, same terminal frame/state within each pair. All source PNG hashes and exact prompts recorded in artifacts/qwen-original-history-diagnostic. Native saved images448x252 front only; enlarging these cannot restore original detail. The repeats have same starting location, not independent maps. Actual inference2.46–4.32s per case, peak4.228GiB; no truncation. All completed.

Both brake repeats under both history durations describe stationary lead as moving and recommend maintaining ego speed. Long history alone does not resolve this case. One lanechange short output literally says steer slightly to the left (to the right), indicating inconsistency. These results do not isolate intrinsic model capability from prompting/resolution/interface; they do refute an unqualified claim that the original model already interprets these inputs reliably.

## Native-resolution and multi-view B2D check
Two independent Town13 heldout recordings were sampled at midpoint anchors62 and128. Original six cameras are1600x900 JPEG; index sets match within each recording but annotation lacks timestamps, so same-index pairing is not independent synchronization proof. Top-down1024x1024 was NOT used. Raw hashes/paths: artifacts/qwen-original-b2d-inputs/manifest.json.
Eight inference comparisons: four front-history448x252 vs896x504; current single front448 vs six current surrounding views448. Exact prompt/state fixed within comparison except required image-context labels. Prompts do not contain scenario family, lead GT or future data. All original model BF16, greedy,repetition_penalty1.1,max180. Inputs/outputs under artifacts/qwen-original-b2d-diagnostic. Peak5.036GiB, percase2.23–11.79s; latency includes shape warmup and output length differences, not a controlled throughput benchmark.
HighwayCutIn:448 says lead stationary while896 says moving. Pending object annotation audit, not counted as improvement by appearance alone.
HardBreak: ego speed0. Front image has extremely close red Dodge and farther blue car. Model names both but single/multiview responses focus blue as lead; higher resolution retains red identification. Recommendation to remain stationary is not itself wrong since ego already0. No calibration or safety claim.
No training/cache/original-Qwen inference processes remained after these runs.

### B2D ground-truth follow-up (labels evaluation only)
Original same-ID historical annotation audit confirms HighwayCutIn anchor62 vehicle22049 (Impala) forward19.550m/right4.140m, speed9.593740m/s; frames56/58/60/62 speeds9.480/9.589/9.469/9.594m/s and world displacement5.745m. Ego9.909055m/s. Thus448 stationary judgment was wrong and896 moving judgment agrees on this single case; near-equal ego/lead speed can make relative visual motion small.
HardBreak128 nearest Dodge15768 forward5.097m, speed-0.00005015m/s (~0), historical position unchanged; ego0. More distant same-lane15767 also stationary, but oncoming15773 speed8.31013m/s and historical displacement4.798m. Not all visible vehicles stationary. Calibration projected boxes provide geometric candidate matches, not occlusion resolution or independent clock synchronization. These labels were never model input.

## Official video-input diagnostic
Prior image-mode prompts ALREADY contained explicit frame offsets (-0.6,-0.4,-0.2,0 or -3,-2,-1,0). Therefore missing textual frame intervals was not established as cause.
Read local transformers Qwen3VLProcessor/VideoProcessor official implementation. Eight saved follow/brake cases rerun as videos, exact same images/state/prompt/greedy/repetition_penalty1.1/max180, do_sample_frames=False. VideoMetadata fps20 and frames_indices[0,4,8,12] or[0,20,40,60]. Official temporal merging encodes timestamps0.1,0.5 or0.5,2.5 seconds. These are start-relative while unchanged prompt offsets are end-relative; both clocks differ by constant offset. Explicit video encoding also changes visual temporal patch aggregation, not only a textual timestamp change.
Results: no consistent correction. Follow repeat0 0.6s wrongly calls moving sedan stopped. Brake repeat0 both spans narrate motion; repeat1 0.6s says stopped or moving slowly but recommends maintaining speed. Six of eight responses reached180token cap with long per-frame narration, so incomplete recommendations must not be scored as finished policies. Per-case2.99–10.57s, peak4.147GiB; all exited, no training/weight changes.
Full inputs/output/processed tokens/metadata: artifacts/qwen-original-video-diagnostic. Reproduce scripts/diagnose_original_qwen_video.py. This small test does not establish temporal encoding as root cause, nor rule out better prompts or larger model capability.
