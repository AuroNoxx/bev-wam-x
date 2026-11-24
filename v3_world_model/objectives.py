"""Stage 3.1 training losses. Inputs here may contain supervision."""
import torch
from scipy.optimize import linear_sum_assignment
from torch.nn import functional as F

from .geometry import DX, DY, H, W, X_MAX, Y_MAX

LANE_POS_WEIGHT = (1., 8., 8., 3.)
STAGE31C_LANE_POS_WEIGHT = (1., 5., 5., 3.)
OCC_POS_WEIGHT = 8.
CLS_WEIGHTS = (1.0, 1.0, 1.5, 2.5, 2.0)
MAP_CLS_WEIGHTS = (1.2, 1.2, 0.8)


def segmentation(logits, target, pos_weight=1., focal=False):
    logits, target = logits.float(), target.float()
    prob = logits.sigmoid()
    weight = logits.new_tensor(pos_weight).reshape(-1, 1, 1)
    loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=weight)
    if focal:
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        pt = prob * target + (1. - prob) * (1. - target)
        alpha = .75 * target + .25 * (1 - target)
        loss = loss + (alpha * (1 - pt).square() * bce).mean()
    prob, target = prob.flatten(2), target.flatten(2)
    dice = (2 * (prob * target).sum(-1) + 1) / (prob.sum(-1) + target.sum(-1) + 1)
    return loss + (1 - dice).mean()


def object_focal_loss(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return ((1 + 7 * target) * (-torch.expm1(-bce)).square() * bce).mean()


def sigmoid_focal_loss(logits, targets, alpha=0.25, gamma=2.0, norm_pos=None):
    logits, targets = logits.float(), targets.float()
    prob = logits.sigmoid()
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = prob * targets + (1.0 - prob) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    loss = bce * ((1.0 - pt) ** gamma) * alpha_t
    denom = targets.sum().clamp_min(1.0) if norm_pos is None else max(float(norm_pos), 1.0)
    return loss.sum() / denom


def detection(prediction, targets):
    boxes = prediction["boxes"].float()
    cls, obj = prediction["cls_logits"].float(), prediction["obj_logits"].float()
    objectness = torch.zeros_like(obj)
    scale = boxes.new_tensor([10., 4., 2., 4., 2., 2., 1., 1., 8., 8.])
    predicted, truth, logits, classes = [], [], [], []
    # Transfer batched costs once across the accelerator/CPU boundary.
    heading = targets[..., 6:7]
    encoded = torch.cat((targets[..., :6], heading.sin(), heading.cos(), targets[..., 7:9]), dim=-1).float()
    ids = targets[..., 9].long()
    with torch.no_grad():
        geo = torch.cdist(boxes / scale, encoded / scale, p=1)
        cls_cost = -cls.log_softmax(dim=-1).gather(2, ids[:, None].expand(-1, boxes.shape[1], -1))
        costs = (geo + cls_cost - obj.sigmoid()[..., None]).cpu().numpy()
        valid = targets[..., -1].bool().cpu().numpy()
    for b, cost in enumerate(costs):
        cols = valid[b].nonzero()[0]
        if not len(cols):
            continue
        rows, selected = linear_sum_assignment(cost[:, cols])
        cols = cols[selected]
        objectness[b, rows] = 1.
        predicted.append(boxes[b, rows])
        truth.append(encoded[b, cols])
        logits.append(cls[b, rows])
        classes.append(ids[b, cols])
    loss = object_focal_loss(obj, objectness)
    if predicted:
        loss = loss + F.smooth_l1_loss(torch.cat(predicted) / scale, torch.cat(truth) / scale)
        loss = loss + F.cross_entropy(torch.cat(logits), torch.cat(classes))
    # Keep unused heads connected on all-empty batches.
    return loss + (boxes.sum() + cls.sum()) * 0.


def _rotated_aabb_bounds(boxes):
    """Differentiable axis-aligned BEV bounding box [x1, y1, x2, y2] from (..., 10) encoded boxes."""
    x, y = boxes[..., 0], boxes[..., 1]
    l, w = boxes[..., 3], boxes[..., 4]
    s, c = boxes[..., 6].abs(), boxes[..., 7].abs()
    hx = 0.5 * (l * c + w * s)
    hy = 0.5 * (l * s + w * c)
    return x - hx, y - hy, x + hx, y + hy


def aligned_bev_giou(boxes_a, boxes_b):
    """Differentiable paired GIoU on BEV footprints."""
    ax1, ay1, ax2, ay2 = _rotated_aabb_bounds(boxes_a)
    bx1, by1, bx2, by2 = _rotated_aabb_bounds(boxes_b)
    inter = (torch.min(ax2, bx2) - torch.max(ax1, bx1)).clamp_min(0.0) * (
        torch.min(ay2, by2) - torch.max(ay1, by1)
    ).clamp_min(0.0)
    area_a = (ax2 - ax1).clamp_min(1e-4) * (ay2 - ay1).clamp_min(1e-4)
    area_b = (bx2 - bx1).clamp_min(1e-4) * (by2 - by1).clamp_min(1e-4)
    union = (area_a + area_b - inter).clamp_min(1e-4)
    iou = inter / union
    enc_w = (torch.max(ax2, bx2) - torch.min(ax1, bx1)).clamp_min(1e-4)
    enc_h = (torch.max(ay2, by2) - torch.min(ay1, by1)).clamp_min(1e-4)
    area_c = (enc_w * enc_h).clamp_min(1e-4)
    return iou - (area_c - union) / area_c


def _pairwise_bev_giou(boxes_a, boxes_b):
    """Pairwise (B, N, M) BEV GIoU matrix for matching cost."""
    ax1, ay1, ax2, ay2 = (t.unsqueeze(2) for t in _rotated_aabb_bounds(boxes_a))
    bx1, by1, bx2, by2 = (t.unsqueeze(1) for t in _rotated_aabb_bounds(boxes_b))
    inter = (torch.min(ax2, bx2) - torch.max(ax1, bx1)).clamp_min(0.0) * (
        torch.min(ay2, by2) - torch.max(ay1, by1)
    ).clamp_min(0.0)
    area_a = (ax2 - ax1).clamp_min(1e-4) * (ay2 - ay1).clamp_min(1e-4)
    area_b = (bx2 - bx1).clamp_min(1e-4) * (by2 - by1).clamp_min(1e-4)
    union = (area_a + area_b - inter).clamp_min(1e-4)
    iou = inter / union
    enc_w = (torch.max(ax2, bx2) - torch.min(ax1, bx1)).clamp_min(1e-4)
    enc_h = (torch.max(ay2, by2) - torch.min(ay1, by1)).clamp_min(1e-4)
    area_c = (enc_w * enc_h).clamp_min(1e-4)
    return iou - (area_c - union) / area_c


def center_heatmap_loss(hm_logits, gt_boxes):
    """CenterNet modified focal loss with anisotropic vehicle Gaussian targets on 160x120 BEV."""
    logits = hm_logits.float()
    b, num_cls, h, w = logits.shape
    gt = gt_boxes.float()
    col_i = ((Y_MAX - gt[..., 1]) / DY - 0.5).round()
    row_i = ((X_MAX - gt[..., 0]) / DX - 0.5).round()
    in_bounds = (gt[..., 10] > 0.5) & (col_i >= 0) & (col_i < w) & (row_i >= 0) & (row_i < h)
    sigma_r = (0.35 * gt[..., 3] / DX).clamp(min=1.5)
    sigma_c = (0.35 * gt[..., 4] / DY).clamp(min=1.5)
    r_idx = torch.arange(h, device=logits.device, dtype=torch.float32)[None, None, :, None]
    c_idx = torch.arange(w, device=logits.device, dtype=torch.float32)[None, None, None, :]
    gauss = torch.exp(
        -0.5 * (((r_idx - row_i[:, :, None, None]) / sigma_r[:, :, None, None]) ** 2
                + ((c_idx - col_i[:, :, None, None]) / sigma_c[:, :, None, None]) ** 2)
    ) * in_bounds[:, :, None, None].float()

    cls_ids = gt[..., 9].long().clamp(0, num_cls - 1)
    hm_gt = torch.stack([
        (gauss * ((cls_ids == c) & in_bounds)[:, :, None, None].float()).amax(dim=1)
        for c in range(num_cls)
    ], dim=1)

    pred_p = logits.sigmoid().clamp(1e-4, 1.0 - 1e-4)
    pos = hm_gt >= 0.999
    pos_loss = -torch.log(pred_p) * ((1.0 - pred_p) ** 2) * pos.float()
    neg_weight = (1.0 - hm_gt).clamp_min(0.0) ** 4
    neg_loss = -torch.log(1.0 - pred_p) * (pred_p ** 2) * neg_weight * (~pos).float()
    cls_w = logits.new_tensor(CLS_WEIGHTS).reshape(1, num_cls, 1, 1)
    return ((pos_loss + neg_loss) * cls_w).sum() / pos.float().sum().clamp_min(1.0)


def _single_det_v3(layer_pred, encoded, ids, valid_mask):
    boxes = layer_pred["boxes"].float()
    cls, obj = layer_pred["cls_logits"].float(), layer_pred["obj_logits"].float()
    objectness = torch.zeros_like(obj)
    scale = boxes.new_tensor([10., 4., 2., 4., 2., 2., 1., 1., 8., 8.])
    predicted, truth, logits, classes = [], [], [], []
    with torch.no_grad():
        xy_cost = torch.cdist(boxes[..., :2] / 6.0, encoded[..., :2] / 6.0, p=1)
        geo = torch.cdist(boxes / scale, encoded / scale, p=1)
        giou_cost = 1.0 - _pairwise_bev_giou(boxes, encoded)
        cls_cost = -cls.log_softmax(dim=-1).gather(2, ids[:, None].expand(-1, boxes.shape[1], -1))
        costs = (2.0 * xy_cost + geo + 1.5 * giou_cost + cls_cost - obj.sigmoid()[..., None]).cpu().numpy()
        valid = valid_mask.cpu().numpy()
    for b, cost in enumerate(costs):
        cols = valid[b].nonzero()[0]
        if not len(cols):
            continue
        rows, selected = linear_sum_assignment(cost[:, cols])
        cols = cols[selected]
        objectness[b, rows] = 1.0
        predicted.append(boxes[b, rows])
        truth.append(encoded[b, cols])
        logits.append(cls[b, rows])
        classes.append(ids[b, cols])
    loss = sigmoid_focal_loss(obj, objectness, alpha=0.25, gamma=2.0)
    if predicted:
        pm, tm = torch.cat(predicted), torch.cat(truth)
        loss = loss + 1.2 * F.smooth_l1_loss(pm / scale, tm / scale)
        loss = loss + 0.8 * (1.0 - aligned_bev_giou(pm, tm)).mean()
        loss = loss + F.cross_entropy(torch.cat(logits), torch.cat(classes), weight=cls.new_tensor(CLS_WEIGHTS))
    return loss + (boxes.sum() + cls.sum()) * 0.0


def detection_v3(prediction, targets):
    """Multi-layer calibrated detection loss + BEV GIoU + DN-DETR positive/negative denoising."""
    heading = targets[..., 6:7]
    encoded = torch.cat((targets[..., :6], heading.sin(), heading.cos(), targets[..., 7:9]), dim=-1).float()
    ids = targets[..., 9].long().clamp(0, len(CLS_WEIGHTS) - 1)
    valid = targets[..., -1] > 0.5

    aux = prediction.get("aux_objects", ())
    total = _single_det_v3(prediction, encoded, ids, valid)
    layer_weights = (0.5, 0.75)
    for w, layer_pred in zip(layer_weights, aux):
        total = total + w * _single_det_v3(layer_pred, encoded, ids, valid)

    dn = prediction.get("dn_outputs")
    if dn is not None:
        meta = dn["meta"]
        pos_valid, neg_valid = meta["pos_valid"], meta["neg_valid"]
        pos_tgt, pos_cls = meta["pos_targets"], meta["pos_cls"]
        scale = encoded.new_tensor([10., 4., 2., 4., 2., 2., 1., 1., 8., 8.])
        cls_w = encoded.new_tensor(CLS_WEIGHTS)
        dn_weights = (0.5, 0.75, 1.0)
        for dw, dl in zip(dn_weights[:len(dn["layers"])], dn["layers"]):
            all_obj_logits = torch.cat((dl["pos_obj_logits"][pos_valid], dl["neg_obj_logits"][neg_valid]))
            all_obj_tgt = torch.cat((
                torch.ones_like(dl["pos_obj_logits"][pos_valid]),
                torch.zeros_like(dl["neg_obj_logits"][neg_valid]),
            ))
            dn_l = (
                sigmoid_focal_loss(all_obj_logits, all_obj_tgt, alpha=0.25, gamma=2.0,
                                   norm_pos=float(pos_valid.sum().clamp_min(1)))
                if all_obj_logits.numel()
                else dl["pos_obj_logits"].sum() * 0.0
            )
            if pos_valid.any():
                pm = dl["pos_boxes"][pos_valid].float()
                tm = pos_tgt[pos_valid].float()
                dn_l = dn_l + 1.2 * F.smooth_l1_loss(pm / scale, tm / scale)
                dn_l = dn_l + 0.8 * (1.0 - aligned_bev_giou(pm, tm)).mean()
                dn_l = dn_l + F.cross_entropy(dl["pos_cls_logits"][pos_valid].float(), pos_cls[pos_valid], weight=cls_w)
            else:
                dn_l = dn_l + (dl["pos_boxes"].sum() + dl["pos_cls_logits"].sum()) * 0.0
            total = total + dw * dn_l
    return total


def _single_maptr_loss(layer_pred, gt_polys, gt_rev, gt_cls, gt_valid):
    pts = layer_pred["map_polylines"].float()
    cls = layer_pred["map_cls_logits"].float()
    obj = layer_pred["map_obj_logits"].float()
    objectness = torch.zeros_like(obj)
    with torch.no_grad():
        d_fwd = (pts[:, :, None, :, :] - gt_polys[:, None, :, :, :]).abs().mean(dim=(-2, -1))
        d_rev = (pts[:, :, None, :, :] - gt_rev[:, None, :, :, :]).abs().mean(dim=(-2, -1))
        d_min = torch.minimum(d_fwd, d_rev) / 8.0
        cls_cost = -cls.log_softmax(dim=-1).gather(2, gt_cls[:, None].expand(-1, pts.shape[1], -1))
        costs = (d_min + cls_cost - obj.sigmoid()[..., None]).cpu().numpy()
        valid = gt_valid.cpu().numpy()

    matched_pred, matched_gt, matched_gt_rev, matched_cls_pred, matched_cls_gt = [], [], [], [], []
    for b, cost in enumerate(costs):
        cols = valid[b].nonzero()[0]
        if not len(cols):
            continue
        rows, selected = linear_sum_assignment(cost[:, cols])
        cols = cols[selected]
        objectness[b, rows] = 1.0
        matched_pred.append(pts[b, rows])
        matched_gt.append(gt_polys[b, cols])
        matched_gt_rev.append(gt_rev[b, cols])
        matched_cls_pred.append(cls[b, rows])
        matched_cls_gt.append(gt_cls[b, cols])

    loss = sigmoid_focal_loss(obj, objectness, alpha=0.25, gamma=2.0)
    if matched_pred:
        pm = torch.cat(matched_pred)
        gm = torch.cat(matched_gt)
        grm = torch.cat(matched_gt_rev)
        df = (pm - gm).abs().mean(dim=(-2, -1))
        dr = (pm - grm).abs().mean(dim=(-2, -1))
        g_best = torch.where((dr < df)[:, None, None], grm, gm)
        pts_scale = pts.new_tensor([16.0, 8.0])
        pts_loss = F.smooth_l1_loss(pm / pts_scale, g_best / pts_scale)
        dp = pm[:, 1:] - pm[:, :-1]
        dg = g_best[:, 1:] - g_best[:, :-1]
        seg_ok = (dg.norm(dim=-1) > 0.05).float()
        dir_loss = ((1.0 - F.cosine_similarity(dp, dg, dim=-1, eps=1e-5)) * seg_ok).sum() / seg_ok.sum().clamp_min(1.0)
        cls_loss = F.cross_entropy(torch.cat(matched_cls_pred), torch.cat(matched_cls_gt),
                                   weight=cls.new_tensor(MAP_CLS_WEIGHTS))
        loss = loss + 2.0 * pts_loss + 0.6 * dir_loss + cls_loss
    return loss + (pts.sum() + cls.sum()) * 0.0


def maptr_loss(prediction, y):
    gt_polys = y["map_polylines"].float()
    gt_rev = gt_polys.flip(2)
    gt_cls = y["map_classes"].long().clamp(0, len(MAP_CLS_WEIGHTS) - 1)
    gt_valid = y["map_valid"].bool()
    layers = [*prediction.get("aux_map", ()), dict(
        map_polylines=prediction["map_polylines"],
        map_cls_logits=prediction["map_cls_logits"],
        map_obj_logits=prediction["map_obj_logits"],
    )]
    weights = (0.5, 0.75, 1.0) if len(layers) == 3 else tuple([1.0] * len(layers))
    total = 0.0
    for w, layer_pred in zip(weights, layers):
        total = total + w * _single_maptr_loss(layer_pred, gt_polys, gt_rev, gt_cls, gt_valid)
    return total


def _trajectory_loss(p, y):
    path, target = p["trajectory"].float(), y["trajectory"].float()
    weights = torch.linspace(2., 1., path.shape[1], device=path.device)[:, None] * path.new_tensor([1., 2., .5])
    trajectory = (F.smooth_l1_loss(path, target, reduction="none") * weights).mean()
    tangents = path[:, 1:, :2] - path[:, :-1, :2]
    expected = target[:, 1:, :2] - target[:, :-1, :2]
    moving = (expected.norm(dim=-1) > .03).float()
    curvature = 1. - F.cosine_similarity(expected[:, 1:], expected[:, :-1], dim=-1, eps=1e-5)
    turn_weights = torch.cat((torch.ones_like(moving[:, :1]), 1. + 3. * curvature), dim=1)
    direction = 1. - F.cosine_similarity(tangents, expected, dim=-1, eps=1e-5)
    return trajectory + (direction * moving * turn_weights).sum() / moving.sum().clamp_min(1)


def _vector7_loss(p, y):
    valid = y["vector_valid"].float()
    vec_error = F.smooth_l1_loss(p["vectors"].float(), y["vectors"].float(), reduction="none")
    vec = (vec_error.sum(dim=-1) * valid).sum() / valid.sum().clamp_min(1)
    return vec + F.binary_cross_entropy_with_logits(p["vector_exists_logits"].float(), y["vector_exists"].float())


def stage31_losses(p, y):
    return dict(
        lane=segmentation(p["lane_logits"], y["lanes"], LANE_POS_WEIGHT),
        vectors=_vector7_loss(p, y),
        objects=detection(p, y["boxes"]),
        occupancy=segmentation(p["occupancy_logits"], y["occupancy"], OCC_POS_WEIGHT, focal=True)
        + segmentation(p["future_occupancy_logits"], y["future_occupancy"], OCC_POS_WEIGHT, focal=True),
        trajectory=_trajectory_loss(p, y),
    )


def stage31c_losses(p, y):
    return dict(
        lane=segmentation(p["lane_logits"], y.get("lanes_cont", y["lanes"]), STAGE31C_LANE_POS_WEIGHT),
        maptr=maptr_loss(p, y),
        heatmap=center_heatmap_loss(p["center_heatmap_logits"], y["boxes"]),
        objects=detection_v3(p, y["boxes"]),
        vectors=_vector7_loss(p, y),
        occupancy=segmentation(p["occupancy_logits"], y["occupancy"], OCC_POS_WEIGHT, focal=True)
        + segmentation(p["future_occupancy_logits"], y["future_occupancy"], OCC_POS_WEIGHT, focal=True),
        trajectory=_trajectory_loss(p, y),
    )
