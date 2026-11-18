"""Requested SFT losses and World Critic arithmetic; no extra learned critic."""
import torch
from torch.nn import functional as F
from .kinematics import denormalize_controls, integrate


def masked_mean(value, mask):
    mask = torch.broadcast_to(mask.to(value.dtype), value.shape)
    denominator = mask.sum()
    if denominator.item() == 0:
        raise ValueError("No valid supervised support")
    return (value * mask).sum() / denominator


def focal_probability(probability, target, support, gamma=2., alpha=.25):
    p = probability.float().clamp(1e-6,1-1e-6)
    y = target.float()
    pt = torch.where(y > .5,p,1-p)
    weight = torch.where(y > .5,alpha,1-alpha)
    return masked_mean(-weight*(1-pt).pow(gamma)*pt.log(),support)


def sft_loss(intent_logits, intent, velocity, noise, clean_controls, mixed_controls, flow_time,
             v0, trajectory_xy, world_probability, occupancy_target, occupancy_support,
             delta_latent, current_latent, future_ema_latent, latent_support, dt=.1):
    """Future target tensors stay exclusively in this loss-side interface."""
    with torch.autocast(device_type=velocity.device.type,enabled=False):
        velocity = velocity.float()
        s = flow_time.float().view(-1,1,1)
        endpoint = mixed_controls.float()+(1-s)*velocity
        acceleration = denormalize_controls(endpoint)
        trajectory = integrate(acceleration,v0,dt)
        reason = F.cross_entropy(intent_logits.float(),intent)
        fm = F.mse_loss(velocity,clean_controls.float()-noise.float())
        fm = fm+.2*F.smooth_l1_loss(trajectory[...,:2],trajectory_xy.float())
        jerk = acceleration.diff(dim=-2)/dt
        kin = F.huber_loss(jerk,torch.zeros_like(jerk))+(acceleration[...,1].abs()-3).relu().square().mean()
        residual_target = (future_ema_latent.float()-current_latent.float()[:,None]).detach()
        latent_mse = (delta_latent.float()-residual_target).square().mean(2)
        world = sum(focal_probability(world_probability[:,h],occupancy_target[:,h],occupancy_support[:,h])
                    +.1*masked_mean(latent_mse[:,h],latent_support[:,h]) for h in range(3))
        loss=.5*reason+fm+.2*kin+.5*world
    return loss, {"reason":reason,"fm":fm,"kin":kin,"world":world}


def footprint_collision(probabilities, trajectory, dimensions, support, dt=.1):
    """Return sampled occupied risk and coverage; unknown is never zero-padded free.

    probabilities:[B,3,Y,X], trajectory:[B,T,7], dimensions:[B,2] length/width.
    Geometry is fixed to the requested corridor, all computation in FP32.
    """
    with torch.autocast(device_type=trajectory.device.type,enabled=False):
        p=probabilities.float(); tr=trajectory.float()
        indices=torch.tensor([round(h/dt)-1 for h in (1,2,3)],device=tr.device)
        poses=tr[:,indices]
        pattern=tr.new_tensor([[.5,0],[0,0],[-.5,0],[.5,.5],[.5,-.5],[-.5,.5],[-.5,-.5]])
        offsets=dimensions.float()[:,None,:]*pattern[None]
        c=poses[...,2].cos()[...,None]; s=poses[...,2].sin()[...,None]
        x=poses[...,0,None]+c*offsets[:,None,:,0]-s*offsets[:,None,:,1]
        y=poses[...,1,None]+s*offsets[:,None,:,0]+c*offsets[:,None,:,1]
        grid=torch.stack((2*(x+16)/128-1,2*(y+19.2)/38.4-1),-1)
        # Pixel-center support, plus physical boundaries; bilinear edge padding
        # contributes zero support and cannot turn absent coverage into free.
        sampled=F.grid_sample(p.flatten(0,1)[:,None],grid.flatten(0,1)[:,None],align_corners=False).reshape(len(p),3,7)
        known=F.grid_sample(support.float().flatten(0,1)[:,None],grid.flatten(0,1)[:,None],align_corners=False).reshape(len(p),3,7)
        valid=(known >= .999)&(x >= -16)&(x <112)&(y >=-19.2)&(y <19.2)
        sampled=torch.where(valid,sampled,torch.full_like(sampled,float('nan')))
        return sampled.amax(-1).sum(-1),valid.all(-1).all(-1)


def critic_reward(trajectory, probabilities, support, vehicle_dimensions, target_speed,
                  lead_gap, lead_speed, lead_valid, lane_cost, dt=.1):
    """Obstacle motion and lanes must come from actual perception/tracking inputs.

    Invalid coverage produces invalid reward for caller handling; it is not a new
    emergency controller. This function never fabricates a lead vehicle velocity.
    """
    with torch.autocast(device_type=trajectory.device.type,enabled=False):
        tr=trajectory.float(); speed=tr[...,3]
        collision,covered=footprint_collision(probabilities,tr,vehicle_dimensions,support,dt)
        closing=(speed-lead_speed.float()).clamp_min(.1)
        ttc=torch.where(lead_valid,lead_gap.float()/closing,torch.full_like(speed,float('inf')))
        ttc_cost=(2.5-ttc).relu().square().amax(-1)
        progress=(1-(speed-target_speed[:,None]).abs()/target_speed[:,None].clamp_min(5))
        progress=(progress*(ttc>3)).mean(-1)
        jerk=tr[...,4:6].diff(dim=-2).norm(dim=-1)/dt
        dynamics=(tr[...,5].abs()-3).relu().square().amax(-1)+(jerk-8).relu().square().amax(-1)
        reward=progress-10*collision-5*ttc_cost-2*dynamics-lane_cost.float()
        return reward,{"coverage":covered,"collision":collision,"ttc":ttc_cost,"dynamics":dynamics,"progress":progress}


def relative_advantages(rewards):
    if not torch.isfinite(rewards).all():
        raise ValueError("GRPO cannot optimize undefined/unsupported trajectory rewards")
    return (rewards-rewards.mean(-1,keepdim=True))/(rewards.std(-1,keepdim=True,unbiased=False)+1e-6)
