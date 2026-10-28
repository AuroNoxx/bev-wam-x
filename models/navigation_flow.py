"""Navigation-conditioned residual Flow with identical train/sample route prior.

All candidates share the declared map-assisted physical reference. Candidate0
starts at zero noise; exploration comes from the learned residual flow only.
No teacher/traffic truth and no untrained action-specific priors are injected.
"""
import torch
from torch import nn
from models.flow_matcher import sinusoidal_time_embedding


class NavigationFlow(nn.Module):
    def __init__(self, residual_std=None, d_model=256, layers=2, heads=4, feedforward=1024, permission_dim=0):
        super().__init__()
        self.arch = dict(d_model=d_model, layers=layers, heads=heads, feedforward=feedforward, permission_dim=permission_dim)
        self.history = nn.Linear(4*256, d_model)
        self.ego = nn.Linear(8, d_model)
        self.route = nn.Linear(3, d_model)
        self.route_position = nn.Parameter(torch.zeros(1, 8, d_model))
        self.vehicle = nn.Linear(4, d_model)
        self.permissions = nn.Linear(permission_dim, d_model) if permission_dim else None
        self.waypoint = nn.Linear(4, d_model)
        self.time = nn.Linear(32, d_model)
        self.position = nn.Parameter(torch.randn(1, 10, d_model)*.02)
        self.layers = nn.ModuleList([nn.TransformerDecoderLayer(d_model=d_model, nhead=heads,
            dim_feedforward=feedforward, dropout=.1, activation='gelu', batch_first=True,
            norm_first=True) for _ in range(layers)])
        self.norm = nn.LayerNorm(d_model); self.out = nn.Linear(d_model, 4)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        std = torch.ones(10, 4) if residual_std is None else torch.as_tensor(residual_std, dtype=torch.float32)
        if std.shape != (10, 4) or not torch.isfinite(std).all() or torch.any(std <= 0):
            raise ValueError('Expected finite positive train-only residual scales')
        self.register_buffer('residual_std', std)
        self.register_buffer('vehicle_scale', torch.tensor([5., 2., 3., 1.]))

    def context(self, history, ego, route_points, route_valid, vehicle, maneuver_allowed=None):
        b = len(history)
        if history.shape != (b, 4, 64, 256) or ego.shape != (b, 4, 2):
            raise ValueError('Expected declared observation history')
        if route_points.shape != (b, 8, 2) or route_valid.shape != (b, 8) or vehicle.shape != (b, 4):
            raise ValueError('Navigation/vehicle shape mismatch')
        clean = torch.where(route_valid[..., None], route_points/40., 0.)
        nav = self.route(torch.cat([clean, route_valid[..., None].to(history.dtype)], -1))+self.route_position
        spatial = self.history(history.permute(0, 2, 1, 3).reshape(b, 64, 4*256))
        tokens = [spatial, self.ego(ego.flatten(1))[:, None], nav,
                  self.vehicle(vehicle/self.vehicle_scale)[:, None]]
        if self.permissions is not None:
            if maneuver_allowed is None or maneuver_allowed.shape != (b, self.arch['permission_dim']):
                raise ValueError('Explicit static maneuver observations are required')
            tokens.append(self.permissions(maneuver_allowed.to(history.dtype))[:, None])
        return torch.cat(tokens, 1)

    def vector_field(self, z, time, context):
        x = self.waypoint(z)+self.position+self.time(sinusoidal_time_embedding(time, 32))[:, None]
        for layer in self.layers:
            x = layer(x, context)
        return self.out(self.norm(x))

    def loss(self, history, ego, route_points, route_valid, vehicle, reference, target, noise=None, time=None,
             maneuver_allowed=None):
        residual = (target-reference)/self.residual_std
        noise = torch.randn_like(residual) if noise is None else noise
        time = torch.rand(len(target), device=target.device) if time is None else time
        z = (1-time[:, None, None])*noise+time[:, None, None]*residual
        predicted = self.vector_field(z, time, self.context(history, ego, route_points, route_valid, vehicle, maneuver_allowed))
        return (predicted-(residual-noise)).square().mean()

    @torch.no_grad()
    def sample(self, history, ego, route_points, route_valid, vehicle, reference,
               candidates=8, steps=10, temperature=.85, seed=0, solver='euler', maneuver_allowed=None):
        if candidates < 1 or steps < 1 or solver not in ('euler', 'midpoint'):
            raise ValueError('Invalid integration settings')
        b = len(history)
        context = self.context(history, ego, route_points, route_valid, vehicle, maneuver_allowed).repeat_interleave(candidates, 0)
        gen = torch.Generator(device=history.device).manual_seed(seed)
        z = torch.randn((b, candidates, 10, 4), generator=gen, device=history.device)*temperature
        z[:, 0] = 0.
        z = z.reshape(b*candidates, 10, 4); dt = 1./steps
        for index in range(steps):
            t = z.new_full((len(z),), index*dt)
            vf = self.vector_field(z, t, context)
            if solver == 'midpoint':
                vf = self.vector_field(z+.5*dt*vf, t+.5*dt, context)
            z = z+dt*vf
        return reference[:, None]+z.reshape(b, candidates, 10, 4)*self.residual_std
