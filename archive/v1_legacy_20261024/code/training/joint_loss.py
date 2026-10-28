"""Masked joint VICReg/action loss, including noisy and recovery trajectories."""
import torch
from models.baseline import masked_trajectory_loss
from training.vicreg import regularization


def joint_loss(output, batch, config):
    action = masked_trajectory_loss(output['trajectory'], batch['trajectory'], batch['valid'], batch['action_mask'])
    zero = action.new_zeros(())
    similarity = variance = covariance = zero
    if config.latent_weight:
        mask = batch['world_valid']
        error = (output['latent'].float()-output['target'].float()).square().mean((-1,-2))
        similarity = (error*mask).sum()/mask.sum().clamp_min(1)
        regs = [regularization(output['current'], config.variance_gamma, config.variance_epsilon)]
        for k in range(4):
            selected = output['target'][:, k][mask[:, k]]
            if len(selected) >= 2:
                regs.append(regularization(selected, config.variance_gamma, config.variance_epsilon))
        variance = torch.stack([value[0] for value in regs]).mean()
        covariance = torch.stack([value[1] for value in regs]).mean()
    latent = config.lambda_sim*similarity+config.lambda_var*variance+config.lambda_cov*covariance
    total = config.latent_weight*latent+config.beta_action*action
    return total, dict(action=action, prediction=similarity, variance=variance, covariance=covariance, latent=latent)
