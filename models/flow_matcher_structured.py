"""Generate whole-trajectory residual coefficients fitted strictly on train data."""
import numpy as np
import torch
from torch import nn

from models.flow_matcher import sinusoidal_time_embedding
from models.flow_matcher_metric import physical_endpoint_losses
from models.flow_matcher_transformer import TransformerKinematicFlowMatcher
from models.joint import constant_turn_trajectory


def fit_trajectory_basis(dataset, dimensions=12):
    """Train-only PCA of standardized 10x4 residuals; never fit on validation."""
    if dataset.split != 'train':
        raise ValueError('Trajectory basis may only be fitted on the train split')
    if not np.all(dataset.action_mask) or not np.all(dataset.waypoint_valid):
        raise ValueError('Basis fitting requires complete valid expert trajectories')
    if not 1 <= dimensions <= 40:
        raise ValueError('Expected 1..40 trajectory coefficients')
    prior = constant_turn_trajectory(torch.from_numpy(dataset.ego)).numpy()
    residual = dataset.targets - dataset.normalizer.trajectory(prior)
    std = residual.std(0).clip(1e-6)
    z = (residual / std).reshape(-1, 40).astype(np.float64)
    mean = z.mean(0); centered = z - mean
    eigenvalues, vectors = np.linalg.eigh(centered.T @ centered / len(centered))
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]; basis = vectors[:, order[:dimensions]]
    # Eigenvector signs do not change PCA; fixing them makes artifacts reproducible.
    for i in range(dimensions):
        if basis[np.argmax(np.abs(basis[:, i])), i] < 0: basis[:, i] *= -1
    return dict(residual_std=std.astype(np.float32), basis=basis.astype(np.float32),
                residual_mean=mean.astype(np.float32),
                coefficient_std=np.sqrt(eigenvalues[:dimensions].clip(1e-8)).astype(np.float32),
                eigenvalues=eigenvalues, explained_variance=float(eigenvalues[:dimensions].sum()/eigenvalues.sum()))


class StructuredFlowMatcher(TransformerKinematicFlowMatcher):
    """Same context/decoder body; flow integrates a 12D whole-trajectory state.

    There is no per-waypoint independent noise and no test-time trajectory repair.
    A small new output projection predicts coefficients jointly from all waypoints.
    """
    def __init__(self, dimensions=12, basis=None, residual_mean=None, coefficient_std=None, **kwargs):
        super().__init__(**kwargs)
        self.dimensions = dimensions
        del self.out_proj
        self.coefficient_head = nn.Linear(10 * self.d_model, dimensions)
        nn.init.zeros_(self.coefficient_head.weight); nn.init.zeros_(self.coefficient_head.bias)
        self.register_buffer('basis', torch.as_tensor(np.eye(40, dimensions) if basis is None else basis, dtype=torch.float32))
        self.register_buffer('residual_mean', torch.as_tensor(np.zeros(40) if residual_mean is None else residual_mean, dtype=torch.float32))
        self.register_buffer('coefficient_std', torch.as_tensor(np.ones(dimensions) if coefficient_std is None else coefficient_std, dtype=torch.float32))
        if self.basis.shape != (40, dimensions) or self.coefficient_std.shape != (dimensions,):
            raise ValueError('Invalid trajectory basis dimensions')
        if not torch.isfinite(self.basis).all() or not (self.coefficient_std > 0).all():
            raise ValueError('Invalid trajectory basis values')

    @property
    def model_arch(self):
        return dict(token_count=self.token_count, token_dim=self.token_dim, d_model=self.d_model,
                    n_heads=self.n_heads, n_layers=self.n_layers, dimensions=self.dimensions,
                    model_type='structured_trajectory_pca')

    def encode_residual(self, residual):
        return ((residual.flatten(-2) - self.residual_mean) @ self.basis) / self.coefficient_std

    def decode_residual(self, coefficients):
        flat = self.residual_mean + (coefficients * self.coefficient_std) @ self.basis.T
        return flat.reshape(*coefficients.shape[:-1], 10, 4)

    def coefficient_velocity(self, coefficients, t, context):
        z_t = self.decode_residual(coefficients)
        h = self.waypoint_proj(z_t) + self.pos_emb + self.time_proj(sinusoidal_time_embedding(t,32))[:,None]
        for layer in self.layers: h = layer(tgt=h, memory=context)
        return self.coefficient_head(self.out_norm(h).flatten(1))

    def compute_loss(self, history, ego, target, action_mask, waypoint_valid,
                     noise=None, t=None, ade_weight=.25, fde_weight=.5, return_parts=False):
        if torch.any(action_mask & ~waypoint_valid.all(-1)):
            raise ValueError('Global trajectory basis requires all 10 waypoints; partial labels are unsupported')
        valid = action_mask[:,None] & waypoint_valid
        prior = self.compute_norm_prior(ego)
        clean = torch.where(valid[...,None], target, prior)
        endpoint = self.encode_residual((clean-prior)/self.residual_std.clamp_min(1e-6))
        source = torch.randn_like(endpoint) if noise is None else noise
        if source.shape != endpoint.shape: raise ValueError('Noise must have shape [batch, dimensions]')
        endpoint = torch.where(action_mask[:,None], endpoint, source)
        time = torch.rand(len(target),device=target.device) if t is None else t
        c_t = (1-time[:,None])*source + time[:,None]*endpoint
        velocity = self.coefficient_velocity(c_t,time,self.extract_context(history,ego))
        error = (velocity-(endpoint-source)).square() * action_mask[:,None]
        flow = error.sum()/(action_mask.sum()*self.dimensions).clamp_min(1)
        estimate = c_t + (1-time[:,None])*velocity
        reconstructed = prior + self.decode_residual(estimate)*self.residual_std
        scale = torch.stack([self.trajectory_scale]*3+[self.trajectory_scale.new_tensor(1.)])
        prediction_si = (reconstructed*self.trajectory_std+self.trajectory_mean)/scale
        truth_si = (clean*self.trajectory_std+self.trajectory_mean)/scale
        ade,fde = physical_endpoint_losses(prediction_si,truth_si,valid)
        loss = flow + ade_weight*ade + fde_weight*fde
        parts = dict(flow=flow,reconstructed_ade_m=ade,reconstructed_fde_m=fde,total=loss)
        return (loss,velocity,parts) if return_parts else (loss,velocity)

    @torch.no_grad()
    def integrate_source(self, history, ego, source, num_steps=10):
        """Source is [batch,candidates,dimensions]; no future/GT is consumed."""
        b,k,_ = source.shape
        c = source.flatten(0,1)
        context = self.extract_context(history,ego).repeat_interleave(k,0)
        for step in range(num_steps):
            t = torch.full((len(c),),step/num_steps,device=c.device,dtype=c.dtype)
            c = c + self.coefficient_velocity(c,t,context)/num_steps
        residual = self.decode_residual(c.reshape(b,k,self.dimensions))
        return self.compute_norm_prior(ego)[:,None] + residual*self.residual_std

    @torch.no_grad()
    def sample(self, history, ego, num_steps=10, num_candidates=1, temperature=1., seed=None, device=None):
        device = history.device if device is None else device
        generator = None if seed is None else torch.Generator(device=device).manual_seed(seed)
        source = torch.randn(len(history),num_candidates,self.dimensions,device=device,
                             dtype=history.dtype,generator=generator)*temperature
        result = self.integrate_source(history,ego,source,num_steps)
        return result[:,0] if num_candidates==1 else result


@torch.no_grad()
def sample_structured_anchors(model, history, ego, anchors, temperature, candidates=8):
    source = torch.stack([torch.randn(candidates,model.dimensions,device=history.device,
                  generator=torch.Generator(device=history.device).manual_seed(42+int(a))) for a in anchors]) * temperature
    return model.integrate_source(history,ego,source)
