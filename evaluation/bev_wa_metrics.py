"""Streaming, label-aware Stage 4 trajectory metrics (CPU / NumPy only)."""
from __future__ import annotations

import math
import numpy as np

OPEN_LOOP_THRESHOLDS = {"val_ade_2s": 0.25, "val_fde_2s": 0.60, "neg_speed_pct": 0.01}


class TrajectoryMetrics:
    """ADE weights valid waypoints; FDE uses only the labelled 2.0 s endpoint.

    ``eligible`` is the dataset action_mask, not inferred from prediction quality.
    Missing labels never become zero error. Nonfinite labelled predictions remain
    in denominators and invalidate their cohort rather than disappearing.
    """

    def __init__(self):
        self.groups = {k: dict(anchors=0, valid_points=0, missing_points=0,
                              endpoints=0, points_1s=0, negative_speeds=0,
                              nonfinite_points=0, error_sum=0.0, error_1s_sum=0.0,
                              endpoint_error_sum=0.0)
                       for k in ("eligible", "all_valid_labels", "ineligible")}

    def update(self, prediction, target, valid, eligible):
        p, y = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
        mask, eligible = np.asarray(valid, dtype=bool), np.asarray(eligible, dtype=bool)
        if p.shape != y.shape or p.ndim != 3 or p.shape[1:] != (10, 4):
            raise ValueError("prediction and target must both have shape [N, 10, 4]")
        if mask.shape != p.shape[:2] or eligible.shape != (len(p),):
            raise ValueError("valid must be [N, 10] and eligible must be [N]")
        errors = np.linalg.norm(p[..., :2] - y[..., :2], axis=-1)
        finite = np.isfinite(p).all(axis=-1) & np.isfinite(y).all(axis=-1)
        for name, select in (("eligible", eligible), ("all_valid_labels", np.ones(len(p), bool)),
                             ("ineligible", ~eligible)):
            g = self.groups[name]
            m = mask & select[:, None]
            end = m[:, -1]
            g["anchors"] += int(select.sum())
            g["valid_points"] += int(m.sum())
            g["missing_points"] += int((~mask[select]).sum())
            g["endpoints"] += int(end.sum())
            g["points_1s"] += int(m[:, :5].sum())
            g["negative_speeds"] += int(((p[..., 2] < 0) & m).sum())
            g["nonfinite_points"] += int((m & ~finite).sum())
            # Index first: invalid NaN/Inf labels must not pollute sums via NaN * 0.
            g["error_sum"] += float(errors[m].sum())
            g["error_1s_sum"] += float(errors[:, :5][m[:, :5]].sum())
            g["endpoint_error_sum"] += float(errors[end, -1].sum())

    def result(self):
        result = {}
        for name, g in self.groups.items():
            def ratio(numerator, denominator):
                return (float(numerator / denominator)
                        if denominator and not g["nonfinite_points"] and math.isfinite(numerator) else None)
            result[name] = {
                **{k: v for k, v in g.items() if not k.endswith("_sum")},
                "val_ade_1s": ratio(g["error_1s_sum"], g["points_1s"]),
                "val_ade_2s": ratio(g["error_sum"], g["valid_points"]),
                "val_fde_2s": ratio(g["endpoint_error_sum"], g["endpoints"]),
                "neg_speed_pct": ratio(100 * g["negative_speeds"], g["valid_points"]),
            }
        primary = result["eligible"]
        return {k: primary[k] for k in ("val_ade_1s", *OPEN_LOOP_THRESHOLDS)} | {
            "metric_contract": "eligible_action_anchors; valid-point-weighted ADE; labelled-2s FDE; unrounded gates",
            "cohorts": result, "open_loop_gate": open_loop_gate(primary),
        }


def open_loop_gate(metrics):
    """Original frozen limits; absent/nonfinite or incomplete primary labels cannot pass."""
    checks = {key: (None if metrics.get(key) is None or not math.isfinite(metrics[key])
                    else metrics[key] <= limit) for key, limit in OPEN_LOOP_THRESHOLDS.items()}
    coverage = (metrics.get("anchors", 0) > 0
                and metrics.get("valid_points") == metrics.get("anchors", 0) * 10
                and metrics.get("endpoints") == metrics.get("anchors")
                and metrics.get("nonfinite_points", 0) == 0)
    return {"thresholds": OPEN_LOOP_THRESHOLDS.copy(), "checks": checks,
            "complete_primary_labels": coverage,
            "passed": bool(coverage and all(value is True for value in checks.values()))}


def format_metric(value, digits=4):
    return "缺失" if value is None or not math.isfinite(value) else f"{value:.{digits}f}"


def render_stage4_report(results):
    """Evidence-only report: no invented physical success, equivalence or latency."""
    lines = ["# 阶段4复评记录", "", "开环原门槛：ADE ≤ 0.25 m；FDE ≤ 0.60 m；负速度 ≤ 0.01%。",
             "主指标仅使用 action_mask 合格锚点；ADE 按有效路点加权，FDE 只取有效2秒终点。",
             "完整分母、缺失标签和不合格样本诊断见 results.json 的 cohorts。",
             "闭环混合场景的路线完成度不用于统一验收；本报告不证明驾驶能力或联合模型额外收益。",
             "A/B/C/D 输入与训练合同是否相同需另行审计，历史结果仅供参照。", ""]
    for group in ("historical_references", "fair_arms", "evaluated_checkpoints"):
        if group not in results:
            continue
        lines.extend([f"## {group}", "", "| 模型 | 种子 | 合格锚点 | 有效路点 | 有效终点 | ADE/m | FDE/m | 负速度/% | 开环门槛 |", "|---|---|---|---|---|---|---|---|---|"])
        for name, arm in results[group].items():
            for seed, row in arm.items():
                if not seed.startswith("seed_"):
                    continue
                cohort = row.get("cohorts", {}).get("eligible", {})
                gate = open_loop_gate(cohort)
                status = "通过" if gate["passed"] else "未通过或缺少证据"
                lines.append(f"| {name} | {seed} | {cohort.get('anchors', '缺失')} | {cohort.get('valid_points', '缺失')} | {cohort.get('endpoints', '缺失')} | {format_metric(row.get('val_ade_2s'))} | {format_metric(row.get('val_fde_2s'))} | {format_metric(row.get('neg_speed_pct'))} | {status} |")
        lines.append("")
    lines.extend(["## 验收边界", "", "阶段4整体尚未验收。开环通过不能替代闭环安全验收、场景分项验证及完整交付。",
                  "闭环分类验收与原始逐步时延应由独立复测记录提供；种子P95均值不是合并P95。", ""])
    return "\n".join(lines)
