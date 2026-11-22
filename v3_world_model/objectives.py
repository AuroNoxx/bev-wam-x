"""Stage 3.1 training losses. Inputs here may contain supervision."""
import torch
from scipy.optimize import linear_sum_assignment
from torch.nn import functional as F


def segmentation(logits, target, focal=False):
    logits, target = logits.float(), target.float()
    prob = logits.sigmoid()
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    if focal:
        pt = prob * target + (1. - prob) * (1. - target)
        alpha = .75 * target + .25 * (1 - target)
        return bce.mean() + (alpha * (1 - pt).square() * bce).mean()
    prob, target = prob.flatten(2), target.flatten(2)
    dice = (2 * (prob * target).sum(-1) + 1) / (prob.sum(-1) + target.sum(-1) + 1)
    return bce.mean() + (1 - dice).mean()


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
    loss = F.binary_cross_entropy_with_logits(obj, objectness)
    if predicted:
        loss = loss + F.smooth_l1_loss(torch.cat(predicted) / scale, torch.cat(truth) / scale)
        loss = loss + F.cross_entropy(torch.cat(logits), torch.cat(classes))
    # Keep unused heads connected on all-empty batches.
    return loss + (boxes.sum() + cls.sum()) * 0.


def stage31_losses(p, y):
    valid = y["vector_valid"].float()
    vec_error = F.smooth_l1_loss(p["vectors"].float(), y["vectors"].float(), reduction="none")
    vec = (vec_error.sum(dim=-1) * valid).sum() / valid.sum().clamp_min(1)
    vec = vec + F.binary_cross_entropy_with_logits(p["vector_exists_logits"].float(), y["vector_exists"].float())

    path, target = p["trajectory"].float(), y["trajectory"].float()
    weights = torch.linspace(2., 1., path.shape[1], device=path.device)[:, None] * path.new_tensor([1., 2., .5])
    trajectory = (F.smooth_l1_loss(path, target, reduction="none") * weights).mean()
    tangents = path[:, 1:, :2] - path[:, :-1, :2]
    expected = target[:, 1:, :2] - target[:, :-1, :2]
    moving = (expected.norm(dim=-1) > .03).float()
    curvature = 1. - F.cosine_similarity(expected[:, 1:], expected[:, :-1], dim=-1, eps=1e-5)
    turn_weights = torch.cat((torch.ones_like(moving[:, :1]), 1. + 3. * curvature), dim=1)
    direction = 1. - F.cosine_similarity(tangents, expected, dim=-1, eps=1e-5)
    trajectory = trajectory + (direction * moving * turn_weights).sum() / moving.sum().clamp_min(1)
    return dict(lane=segmentation(p["lane_logits"], y["lanes"]), vectors=vec,
                objects=detection(p, y["boxes"]),
                occupancy=segmentation(p["occupancy_logits"], y["occupancy"], focal=True) +
                segmentation(p["future_occupancy_logits"], y["future_occupancy"], focal=True),
                trajectory=trajectory)
