"""Small spatial navigation-conditioned MSE baseline over frozen BEV tokens."""
import torch
from torch import nn
from models.joint import constant_turn_trajectory


class NavigationTrajectoryHead(nn.Module):
    def __init__(self, tokens=64, token_dim=256, route_points=8, hidden=512, reference_prior=False, permission_dim=0):
        super().__init__()
        self.arch = dict(tokens=tokens, token_dim=token_dim, route_points=route_points, hidden=hidden,
                         reference_prior=reference_prior, permission_dim=permission_dim)
        self.spatial = nn.Sequential(nn.Linear(4*token_dim, 32), nn.GELU())
        self.route = nn.Sequential(nn.Linear(route_points*3, 128), nn.GELU())
        self.head = nn.Sequential(nn.Linear(tokens*32+128+8+4+permission_dim, hidden), nn.GELU(),
                                  nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 40))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)
        self.register_buffer('scale', torch.tensor([20., 5., 10., 1.]))
        self.register_buffer('vehicle_scale', torch.tensor([5., 2., 3., 1.]))

    def forward(self, history, ego, route_points, route_valid, vehicle, reference=None, maneuver_allowed=None):
        b, t, n, d = history.shape
        if (t, n, d) != (4, self.arch['tokens'], self.arch['token_dim']):
            raise ValueError('Expected four spatial token frames')
        if route_points.shape != (b, self.arch['route_points'], 2) or route_valid.shape != route_points.shape[:2]:
            raise ValueError('Navigation shape mismatch')
        clean_route = torch.where(route_valid[..., None], route_points/40., 0.)
        route = self.route(torch.cat([clean_route, route_valid[..., None].to(history.dtype)], -1).flatten(1))
        content = history.permute(0, 2, 1, 3).reshape(b, n, 4*d)
        spatial = self.spatial(content).flatten(1)
        features = [spatial, route, ego.flatten(1), vehicle/self.vehicle_scale]
        if self.arch['permission_dim']:
            if maneuver_allowed is None or maneuver_allowed.shape != (b, self.arch['permission_dim']):
                raise ValueError('This input profile requires explicitly observed static maneuver permissions')
            features.append(maneuver_allowed.to(history.dtype))
        residual = self.head(torch.cat(features, -1))
        if self.arch['reference_prior']:
            if reference is None:
                from planning.navigation_reference import navigation_reference
                prior = navigation_reference(ego.detach().cpu().numpy(), route_points.detach().cpu().numpy(),
                                             route_valid.detach().cpu().numpy(), vehicle.detach().cpu().numpy())
                prior = torch.from_numpy(prior).to(history.device)
            else:
                if reference.shape != (b, 10, 4):
                    raise ValueError('Reference shape mismatch')
                prior = reference
        else:
            prior = constant_turn_trajectory(ego.float())
        return prior+residual.reshape(b, 10, 4).float()*self.scale

