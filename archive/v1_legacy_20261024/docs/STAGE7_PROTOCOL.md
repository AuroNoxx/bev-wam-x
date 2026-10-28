# Stage 7 Protocol: Flow Matching Candidate Generation and Risk-Aware Trajectory Planning

> 2026-10-22版本说明：本文保留原阶段7实验协议、追加条款与历史门槛，不能用于回写旧结果。当前阶段7补充范围已更新为导航条件融合驾驶，见 [GOAL.md](../GOAL.md) 和 [改造方案](INTEGRATED_DRIVING_PLAN.md)。新实验须另行冻结协议。本文旧15–30M预算、板卡定位及“无需验收视频”等历史条款不覆盖当前AGENTS.md约定：当前50M–100M预算、完整决策P95≤50ms、原生ROCm、每次验收需视频，无Orin-X部署要求。旧head-only 30ms与报告50ms差异须显式区分，不能改判旧门槛。

## 1. Context and Mission Boundary

- **Authorized Stage 7 Step 1 (2026-10-19):** Authorized by user instruction with Gemini implementing model/data components and Codex supervising planning and acceptance.
- **Architectural & Hardware Discipline:**
  - Strict parameter budget: **15–30M total parameters**, including frozen Stage 4 Joint encoder and Stage 6 Action-Conditioned World Predictor.
  - Frozen modules:
    - Stage 4 Joint Model (`models/joint.py` checkpoint `artifacts/restart_stage4/joint_001/best.pt`).
    - Stage 6 Action-Conditioned World Predictor (`models/action_conditioned.py` checkpoint `artifacts/restart_stage6/runs/conditioned_20261019/best.pt`).
  - PyTorch ROCm execution only; functional SDPA MATH backend; no CUDA dependencies, flash-attn, xformers, Triton, custom CUDA operators, or `torch.compile`.
  - Retention of all previous data, reports, failed runs, and MREs. No new acceptance video required for this phase.

---

## 2. Scientific Questions and Hypotheses

1. **Trajectory Diversity & Feasibility:** Can a Linear-interpolation Conditional Flow Matching generator produce diverse, kinematically smooth, and road-compliant candidate trajectories from Gaussian noise, conditioned on BEV representations, achieving non-oracle sample 1 ADE/FDE $\le 1.25 \times \text{MSE}$ while maintaining pairwise diversity $\ge 0.1\text{m}$?
2. **Latent World Risk Anticipation:** Does conditioning a risk prediction head on Stage 6 action-conditioned future latent predictions yield superior crash and off-road prediction accuracy (AUROC $\ge 0.75$, AUPRC $>$ test prevalence, Brier score relative improvement $\ge 5\%$ over unconditioned control) compared to an identical risk head without world model rollouts?
3. **Closed-Loop Decision Benefit:** Does multi-candidate generation combined with world-model-guided risk scoring achieve statistically significant improvements in route completion ($\Delta \ge 0.02$, 95% CI lower bound $> 0$) and non-increasing collision/offroad rates over:
   - Arm 1: Deterministic MSE trajectory regression (Stage 4 baseline).
   - Arm 2: Flow Matching with geometric/kinematic scoring only (no world risk).
   - Arm 3: Flow Matching with geometric/kinematic scoring plus Stage 6 latent world risk prediction.

---

## 3. Model Architecture and Interfaces

### 3.1 Conditional Flow Matching Trajectory Head
- **Target:** Normalized 10 waypoints at $\Delta t = 0.2\text{s}$ ($T = 2.0\text{s}$), $x_1 \in \mathbb{R}^{10 \times 4}$ in ego coordinates $(x, y, v, \text{heading})$.
- **Linear-Interpolation Formulation:**
  $$x_t = (1 - t) x_0 + t x_1, \quad t \in [0, 1], \quad x_0 \sim \mathcal{N}(0, I)$$
  Target velocity field:
  $$u_t(x_t | x_1) = x_1 - x_0$$
  Invalid target coordinates are sanitized to $x_0$ prior to interpolation, ensuring invalid points cannot corrupt $x_t$.
  Loss is normalized by the total valid coordinate channels ($\text{valid\_points} \times 4$).
- **Supervision Rule:** Supervised strictly on schema-2 expert demonstrations (`future_trajectory` with `action_mask` and `valid`). Handcrafted candidate branches must **never** be used as demonstration targets.
- **Inference Integration:** Fixed Euler ODE integration ($N_{\text{steps}} = 10$) from sampled Gaussian noise $x_0$, batched across $K = 8$ candidate samples per decision step. Fixed noise seeds ($42 + \text{anchor}$) used for strict reproducibility.
- **Frozen Training Hyperparameters:**
  - 25 epochs, batch size 64, learning rate $2 \times 10^{-4}$, AdamW, weight decay $10^{-4}$, cosine schedule, gradient clip 5.0.
  - Subsampled dataset: train 20000, val 2048, test 2048 (eligible anchors with `action_mask == 1` and all 10 future waypoints valid, subsampled with seed 20261021).
  - Two training seeds: `20261021` and `20261022`.
  - Checkpoint selection: minimum fixed-target validation loss.

### 3.2 Action-Conditioned Risk Head
- **Inputs:** Candidate commanded trajectory $a \in \mathbb{R}^{10 \times 4}$, ego history, and Stage 6 predicted future latents $z_{\text{pred}} \in \mathbb{R}^{4 \times 64 \times 256}$.
- **Causal Constraint:** Strictly forbidden to input ground-truth future states or BEVs.
- **Matched Control Baseline:** Identical architecture risk head with $z_{\text{pred}}$ zeroed.
- **Outputs & Censoring:**
  - Multi-task binary heads: $P(\text{crash}) \in [0, 1]$ and $P(\text{out\_of\_road}) \in [0, 1]$.
  - Censoring Rule: $\text{label\_valid} = \text{event\_seen} \lor \text{full\_2s\_completed}$.
  - Early terminations from other causes censor unobserved events.
- **Frozen Training Hyperparameters:**
  - 50 epochs, batch size 64, learning rate $1 \times 10^{-3}$, AdamW, weight decay $10^{-4}$, cosine schedule.
  - Two training seeds matching world vs zero-world controls.
  - Selected checkpoint: minimum validation masked BCE loss.
  - Decision thresholds ($\tau_{\text{crash}}, \tau_{\text{offroad}}$): selected by maximum Youden's J ($J = \text{TPR} - \text{FPR}$) on the validation set, with ties broken by the lower threshold. Strictly forbidden to tune thresholds on the test set.

---

## 4. Risk Data Collection Protocol

### 4.1 Native Simulator Supervision vs Heuristics
- Native MetaDrive APIs (`info["crash"]`, `info["out_of_road"]`, `info["arrive_dest"]`) and typed crash events (`crash_vehicle`, `crash_object`, `crash_building`, `crash_human`, `crash_sidewalk`) provide true supervision.
- Candidate commands are strictly $10 \times 4$ trajectories executed by the unmodified `TrajectoryTracker`. Adding hidden controller actions during execution is forbidden.

### 4.2 Roadside Relative Candidate Generation
- To ensure trajectories start smoothly from the vehicle's actual lateral position:
  $$\text{observed\_lateral\_left} = (\mathbf{p}_{\text{car}} - \mathbf{p}_{\text{route}}(0)) \cdot \vec{n}_{\text{left}}$$
  $$\text{lateral}(t) = \text{observed\_lateral\_left} + \text{offset} \cdot h(t)$$
  $$\mathbf{p}(t) = \mathbf{p}_{\text{route}}(s(t)) + \text{lateral}(t) \cdot \vec{n}_{\text{left}}(t)$$
  This avoids MetaDrive right-positive sign ambiguities and allows roadside vehicles to execute valid recovery or extreme offroad commands.
- 13 Candidate Specifications:
  `keep` ($0\text{m}, 7\text{m/s}$), `slow` ($0\text{m}, 3\text{m/s}$), `fast` ($0\text{m}, 12\text{m/s}$), $\pm 6\text{m}$ ($7, 12, 3\text{m/s}$), $\pm 12\text{m}$ ($7, 12\text{m/s}$).

### 4.3 Alternating Prefix Contexts
- Even seed indices ($0, 2, 4 \dots$): Standard center context (expert driving up to `anchor_step`).
- Odd seed indices ($1, 3, 5 \dots$): Right-turn roadside context (expert driving up to $\text{anchor\_step} - 10$, followed by 10 steps of roadside maneuver $\text{offset} = -12\text{m}, v = 12\text{m/s}, t_{\text{shift}} = 0.8\text{s}$ before reaching anchor).
- All 13 candidates under an anchor context replay the exact saved prefix actions from reset with `force_destroy=True`, verifying state error $\le 10^{-5}$ and BEV diff $= 0$.
- Any unexpected prefix termination saves raw MRE and halts with failed manifest.

### 4.4 Split Isolation
- Disjoint map and seed sets:
  - **Train:** Maps `('S', 'C', 'SC', 'CS')`, seeds starting at $710000 + \text{map\_idx} \times 10000$, 8 seeds per map.
  - **Validation:** Maps `('CC', 'SS')`, seeds starting at $810000 + \text{map\_idx} \times 10000$, 4 seeds per map.
  - **Test:** Maps `('SCS', 'CSC')`, seeds starting at $910000 + \text{map\_idx} \times 10000$, 4 seeds per map.
- Anchor prefix steps: $(20, 50, 80, 110)$.

---

## 5. Candidate Trajectory Planning and Scoring

- Planning evaluates $K = 8$ candidates from the conditional Flow Matching model.
- Composite cost:
  $$J(a_i) = w_{\text{progress}} J_{\text{progress}} + w_{\text{speed}} J_{\text{speed}} + w_{\text{smooth}} J_{\text{smooth}} + w_{\text{static}} J_{\text{static}} + w_{\text{dynamic}} J_{\text{dynamic}} + w_{\text{risk}} J_{\text{risk}}$$
- Static geometry: checked against current BEV footprint. Dynamic actors: extrapolated from observable history. World risk: evaluated by risk head conditioned on Stage 6 predicted latents.
- Pruning: candidates with $P(\text{crash}) > \tau_{\text{crash}}$ or $P(\text{out\_of\_road}) > \tau_{\text{offroad}}$ are pruned.
- Specific scoring weights and geometric formulas are frozen in the deployment-scoring section below and `configs/planner7.py`.

---

## 6. Formal Scientific Acceptance Gates

### 6.1 Flow Matching Generation Gate
- Non-oracle candidate 1 ADE and FDE on test set $\le 1.25 \times \text{MSE baseline}$.
- Average pairwise Euclidean position distance $\ge 0.1\text{m}$.
- Kinematic feasibility: $\ge 95\%$ of all candidate points satisfy speed $\in [0, 15\text{m/s}]$, $|a| \le 8\text{m/s}^2$, and heading deviation $\le 0.35\text{rad}$ on moving points.
- Head-only latency P95 $\le 30\text{ms}$.

### 6.2 Risk Prediction Gate
- Test sample sufficiency: Each event category (crash, out_of_road) must have $\ge 20$ positive and $\ge 20$ negative occurrences across $\ge 4$ test seeds.
- AUROC $\ge 0.75$ for both crash and out-of-road events on held-out test data.
- AUPRC strictly greater than test prevalence.
- Brier score strictly lower than constant train prevalence baseline.
- World Model vs Zero-World Control: Relative Brier score improvement $\ge 5\%$, and paired seed-cluster difference 95% bootstrap CI upper bound $< 0$.

### 6.3 Closed-Loop 3-Arm Driving Gate
- Formal cohort: 2 test maps (`SCS`, `CSC`) $\times$ 4 seeds per map $= 8$ seeds total.
- Independent closed-loop seeds: `1010000` and `1020000` (strictly disjoint from risk test seeds).
- 3 arms $\times$ 2 training seeds $\times$ 8 test episodes $= 48$ formal closed-loop episodes.
- Max 1000 control steps ($\Delta t = 0.1\text{s}$) or actual simulator termination.
- Full planner (Arm 3) route completion relative to both controls (Arm 1 and Arm 2): improvement $\ge 0.02$ with 95% bootstrap CI lower bound $> 0$.
- Collision and off-road rates must not be higher than controls.
- Paired bootstrap: 2000 resamples clustered by seed. If no statistically significant improvement is achieved, declare failure honestly.

## Frozen deployment scoring (before any Stage 7 closed-loop test)

2026-10-19: user authorizes Codex to execute remaining work while Relay permissions are investigated. `configs/planner7.py`, `planning/candidates.py` and `planning/policy7.py` define the deployed observation-only planner. No Flow retraining or selection-rule tuning is performed after inspecting the fixed open-loop test results.

For each of eight raw Flow samples, cost terms are: negative final ego-forward distance /20m; mean squared speed deviation from 7m/s divided by 7²; mean |v × heading-rate| /8; mean absolute heading-rate /.35; current-BEV footprint non-road fraction plus twice static-boundary fraction (outside raster costs 1); fraction of horizons intersecting constant-velocity observed actor circles; predicted crash and off-road probabilities. Weights in that order are 1,.5,.2,.1,5,5,10,10. Current ego-covered cells and actor-occluded road are treated as unknown, not filled with imaginary road; visibility loss has an extra .5-fraction penalty. Footprints use known vehicle dimensions and predicted headings. Actor centroids are matched mutually between actual oldest/current BEV frames after observed ego-pose compensation; unmatched actors use zero velocity and additional .5m margin. Other margins are .5m + .25m/s × horizon. Connected-component matching and constant velocity are approximations, not a learned future scene decoder.

Reject any sample with nonfinite fields, speed outside [0,15]m/s, waypoint acceleration over 8m/s², or moving-segment heading/position disagreement over .35rad. Full planning additionally rejects probabilities above validation-selected thresholds. Remaining samples minimize the fixed cost, ties select the lowest candidate ID. If none remains, `selected_id=STOP` executes a deterministic 4m/s² straight braking trajectory through the same tracker. This fallback is explicitly rule-generated and is never counted as a Flow-generated candidate or as evidence of successful planning. All rejected candidates and scores remain visible. Both geometric and full arms use identical raw noise streams and geometric/kinematic rules; their state distributions may diverge after executing different decisions.

Risk prediction from the stress cohort may fail scientific validation. Full planning is then an experimental ablation, not a validated safety system. Failure of generation/risk/planning gates is reported independently; the simpler frozen MSE policy remains retained. Stage 8 cannot be accepted from these experiments. Latency measured alongside collection is labeled shared-load; final timing limitations are disclosed. Closed-loop subsequent latent targets come from the actual receding-horizon rollout, and are not the fixed-command counterfactual outcomes used for risk supervision.

## Pre-training data correction audit (2026-10-20)

Independent review found that the collector's context file stored placeholder zeros for ego history before calculating it. No formal risk model was trained on that placeholder. Preserve these original context/branch files and their manifest hashes. `scripts/recover_stage7_motion.py` replays each saved prefix from the original reset seed and checks four BEV frames byte-for-byte, observed vehicle states, and the complete vehicle scene within 1e-5. It saves five actual observed states in separate immutable sidecars; four speeds/10 and four wrapped backward heading differences/.1s reproduce the Stage 6 motion convention exactly. No subsequent branch observation or test event informs this correction. Formal feature caching requires completed recovery and matching original/sidecar hashes. Earlier zero-history cache/training smoke artifacts are explicitly superseded and retained as defective smoke evidence.

The actual Flow architecture is the checkpoint `model_arch` and frozen source: per-spatial-token temporal linear fusion and a two-hidden-layer vector-field MLP. The initial config's unused `flow_num_layers/flow_num_heads` do not instantiate a Transformer. Risk classification has three Linear layers. The head budget is measured from actual parameters, not config labels.

## Native semantic replay repair (2026-10-20)

Strict replay halted at `train_CS_740007_step50` and `test_CSC_920000_step110`: one boundary-channel pixel changed while all vehicle states, float64 poses, and camera matrices matched. Native RGB at the CSC pixel alternated exactly between SIDEWALK `(244,35,232)` and TERRAIN `(152,251,152)`. Fresh engine resets and a 24-bit depth request did not eliminate it. This supports an order-dependent overlapping-surface semantic classification issue, rather than a dynamics mismatch or label issue.

Camera-only fixed-bin ordering (terrain, road, crosswalk, lane line, sidewalk, other surfaces) eliminated all replay differences in 16 repeated trials at each failed context. Native color states, geometry, physics, depth tests, sensor dimensions and calibration are retained. This is an explicit renderer correction with source diffs and bounded before/after MREs, not a relaxed acceptance tolerance or masked pixel. Default behavior of earlier stages stays unchanged; Stage 7 continuation and all three formal closed-loop arms opt in. No claim is made that the correction repairs every possible renderer nondeterminism.

Original failed partitions remain unchanged. New continuation partitions copy only complete contexts and recompute counts from their complete 13-branch metadata, then collect every remaining context in the original frozen cohort. `semantic_order` is recorded per context, including copied originals. Motion recovery replays each under its original renderer version. Feature caching accepts only the two explicit source versions in `codex_review/determinism_repair.json`; it still enforces the full exact map/seed/anchor cohort, raw hashes, and zero prefix-BEV difference. The mixed renderer versions are disclosed as a limitation. This repair does not tune risk labels, model hyperparameters, or planning thresholds based on test outcomes.
