"""Geometric and photometric augmentation, supervision, and aggregate metrics."""
import math

import torch
from torch.nn import functional as F

from .phase2_model import bev_grid


def change_frame(points, dy, yaw):
    shape = (len(points),) + (1,)*(points.ndim-2)
    c, s = yaw.cos().reshape(shape), yaw.sin().reshape(shape)
    x, y = points.unbind(-1)
    y = y-dy.reshape(shape)
    return torch.stack((c*x+s*y, -s*x+c*y), -1)


def rejoin_path(path, dy, yaw):
    batch = len(path)
    lead = torch.zeros_like(path[:, :1])
    local = change_frame(torch.cat((lead, path), 1), dy, yaw)
    distance = 1.5*torch.arange(41, dtype=path.dtype, device=path.device)
    u = (distance/12.).clamp(0, 1)[None, :, None]
    # Cubic Hermite correction re-anchors to (0,0) and a forward tangent,
    # with zero displacement and slope at the original path's 12 m station.
    initial = -local[:, :1]
    tangent = (local[:, 1:2]-local[:, :1])/1.5
    target = torch.cat((torch.ones_like(tangent[..., :1]), torch.zeros_like(tangent[..., :1])), -1)
    local = local+(2*u**3-3*u**2+1)*initial+(u**3-2*u**2+u)*12*(target-tangent)
    span = torch.linalg.vector_norm(local[:, 1:]-local[:, :-1], dim=-1).clamp_min(1e-5)
    arc = torch.cat((torch.zeros_like(span[:, :1]), span.cumsum(1)), 1)
    # The augmented target stays within measured support when the new approach is shorter.
    query = (distance[1:][None]*torch.minimum(arc[:, -1:], torch.full_like(arc[:, -1:], 60.))/60).contiguous()
    i = torch.searchsorted(arc.contiguous(), query, right=True).clamp(1, 40)
    lower, upper = arc.gather(1, i-1), arc.gather(1, i)
    a, b = (local.gather(1, index[..., None].expand(batch, 40, 2)) for index in (i-1, i))
    return a+(b-a)*((query-lower)/(upper-lower).clamp_min(1e-5))[..., None]


def make_batch(batch, augment, axes):
    lidar, nav, cmd = (batch[k] for k in ('lidar','nav','cmd'))
    native='front' in batch
    rgb=(batch['front'],*batch['sides'].unbind(1)) if native else batch['rgb']
    path, lines, valid, exists = (batch[k] for k in ('path','lines','valid','exists'))
    device = lidar.device
    bits = torch.arange(7, -1, -1, dtype=torch.uint8, device=device)
    raster = ((batch['raster_bits'][..., None] >> bits) & 1).flatten(-2)
    grid, gain, contrast = None, None, None
    if augment:
        n = len(lidar)
        random = torch.rand(n, 6, device=device)
        mirror = random[:, 0] < .5
        cam = [0, 2, 1]
        order = [0, 2, 1, 4, 3, 6, 5]
        if native:
            rgb=tuple(torch.where(mirror[:,None,None,None],rgb[j].flip(-1),rgb[i]) for i,j in enumerate(cam))
        else:
            rgb = torch.where(mirror[:, None, None, None, None], rgb[:, cam].flip(-1), rgb)
        lidar = torch.where(mirror[:, None, None, None], lidar.flip(-1), lidar)
        raster = torch.where(mirror[:, None, None, None], raster.flip(-1), raster)
        reflect = torch.stack((torch.ones_like(random[:, 0]), 1.-2.*mirror.float()), -1)
        nav, path = nav*reflect[:, None], path*reflect[:, None]
        lines = torch.where(mirror[:, None, None, None], lines[:, order], lines)*reflect[:, None, None]
        valid = torch.where(mirror[:, None, None], valid[:, order], valid)
        exists = torch.where(mirror[:, None], exists[:, order], exists)
        cmd = torch.where(mirror[:, None], cmd[:, cam], cmd)
        perturb = random[:, 1] < .30
        dy = (2*random[:, 2]-1)*.7*perturb
        yaw = (2*random[:, 3]-1)*math.radians(5)*perturb
        grid = bev_grid(dy[:,None,None], yaw[:,None,None], *axes)
        lidar = F.grid_sample(lidar.float(),grid,mode='nearest',align_corners=False).to(lidar.dtype)
        raster = F.grid_sample(raster.float(),grid,mode='nearest',align_corners=False)
        nav = change_frame(nav,dy,yaw)
        lines = change_frame(lines,dy,yaw)
        path = torch.where(perturb[:, None, None], rejoin_path(path,dy,yaw), path)
        gain = (.85+.30*random[:,4]).reshape(n,1,1,1,1)
        contrast = (.8+.4*random[:,5]).reshape(n,1,1,1,1)
    inputs = (rgb,lidar,nav,cmd,grid,gain,contrast)
    return inputs, dict(path=path,lines=lines,valid=valid,exists=exists,mask=raster[:,:4],occ=raster[:,4:])


def multitask_loss(outputs, target, perception_priority=False):
    mask, occ = outputs['mask'].float(), outputs['occ'].float()
    mask_true, occ_true = target['mask'].float(), target['occ'].float()
    mask_weights = mask.new_tensor([1,8,8,3] if perception_priority else [1,6,8,2]).reshape(4,1,1)
    channel_weights = mask.new_tensor([1.,2.,1.,2.] if perception_priority else [1.,1.,1.,1.])
    terms = F.binary_cross_entropy_with_logits(mask,mask_true,pos_weight=mask_weights,reduction='none')
    mask_bce = (terms.mean((0,2,3))*channel_weights).sum()/channel_weights.sum()
    prob = mask.sigmoid()
    inter = (prob*mask_true).sum((0,2,3))
    dice = (2*inter+1)/((prob+mask_true).sum((0,2,3))+1)
    mask_dice = 1.-(dice*channel_weights).sum()/channel_weights.sum()
    occ_bce = F.binary_cross_entropy_with_logits(occ,occ_true,pos_weight=occ.new_tensor(30. if perception_priority else 10.))
    per = F.binary_cross_entropy_with_logits(occ,occ_true,reduction='none')
    pt = torch.exp(-per)
    focal = ((.25+.5*occ_true)*(1.-pt).square()*per).mean()
    valid = target['valid'].unsqueeze(-1)
    line_l1 = F.smooth_l1_loss(outputs['lines'],target['lines'],reduction='none')
    line = (line_l1*valid).sum()/(2*valid.sum().clamp_min(1))
    existence = F.binary_cross_entropy_with_logits(outputs['exists'].float(), target['exists'].float())
    path_raw = F.smooth_l1_loss(outputs['path'], target['path'],reduction='none')
    station = 1.5*torch.arange(1,41,device=path_raw.device)
    weights = (1.+(station<=24).float())[:,None]*path_raw.new_tensor([1.,4.])[None]
    path = (path_raw*weights).mean() / weights.mean()
    map_scale, occ_scale = (2.,3.) if perception_priority else (1.,1.)
    return 4.*path + .6*line + .2*existence + map_scale*(mask_bce+mask_dice) + occ_scale*(occ_bce+focal)


def validation_totals(outputs, targets):
    lat = (outputs['path'][..., 1]-targets['path'][..., 1]).abs()
    val = targets['valid']
    length = torch.linalg.vector_norm(outputs['lines']-targets['lines'], dim=-1)
    blocks = [lat[:, :16].sum()[None], lat[:, 16:].sum()[None],
              lat.new_tensor([len(lat)*16,len(lat)*24]), (length*val).sum((0,2)), val.sum((0,2))]
    for key in ('mask','occ'):
        pred, truth = outputs[key]>0, targets[key]>0
        blocks += [(pred & truth).sum((0,2,3)), (pred | truth).sum((0,2,3))]
    return torch.cat(blocks)


def metrics_from_totals(totals):
    t = totals.detach().cpu().numpy()
    avg = lambda x,y: [float(a/b) if b else None for a,b in zip(x,y)]
    line = avg(t[4:11],t[11:18])
    mask, occ = avg(t[18:22],t[22:26]), avg(t[26:31],t[31:36])
    mean = lambda xs: sum(v for v in xs if v is not None)/max(1,sum(v is not None for v in xs))
    return dict(lateral_near_m=float(t[0]/t[2]), lateral_far_m=float(t[1]/t[3]),
                map_line_mean_m=mean(line), map_line_per_query_m=line, map_miou=mean(mask),
                map_iou=mask, occupancy_miou=mean(occ),occupancy_iou=occ)
