"""VICReg on the actual encoder tokens, with fixed-position content centering."""
import torch
import torch.nn.functional as F


def content_covariance(z):
    if z.ndim != 3 or len(z) < 2:
        raise ValueError('Content covariance requires [batch>=2, tokens, channels]')
    z = z.float()
    # Constant positional embeddings cannot satisfy content variance/rank.
    centered = z - z.mean(0, keepdim=True)
    flat = centered.flatten(0, 1)
    return flat.T @ flat / (z.shape[1] * (z.shape[0] - 1))


def regularization(z, gamma=1., epsilon=1e-4):
    covariance = content_covariance(z)
    variance = F.relu(gamma - torch.sqrt(covariance.diagonal() + epsilon)).mean()
    off_diagonal = covariance - torch.diag_embed(covariance.diagonal())
    decorrelation = off_diagonal.square().sum() / covariance.shape[0]
    return variance, decorrelation


def latent_loss(prediction, target, current, config):
    similarity = F.mse_loss(prediction.float(), target.float())
    var_a, cov_a = regularization(current, config.variance_gamma, config.variance_epsilon)
    var_b, cov_b = regularization(target, config.variance_gamma, config.variance_epsilon)
    variance, covariance = (var_a + var_b) / 2, (cov_a + cov_b) / 2
    total = config.lambda_sim * similarity + config.lambda_var * variance + config.lambda_cov * covariance
    return total, dict(prediction=similarity, variance=variance, covariance=covariance)

