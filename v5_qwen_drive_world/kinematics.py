"""FP32 control normalization, differentiable integration and Euler Flow."""
import torch


def denormalize_controls(u):
    with torch.autocast(device_type=u.device.type, enabled=False):
        u = u.float().clamp(-1, 1)
        return torch.stack((u[..., 0] * 6 - 2, u[..., 1] * 3.5), -1)


def normalize_controls(a):
    with torch.autocast(device_type=a.device.type, enabled=False):
        a = a.float()
        return torch.stack(((a[..., 0] + 2) / 6, a[..., 1] / 3.5), -1)


def integrate(controls, v0, dt=0.1, v_floor=5.0, v_max=35.0):
    """Return [..., T, (x,y,yaw,v,ax,ay,kappa)], current fixed FLU.

    This discrete kinematic convention is not a dynamic tyre/slip guarantee.
    """
    with torch.autocast(device_type=controls.device.type, enabled=False):
        a = controls.float(); v = v0.float()
        x = torch.zeros_like(v); y = torch.zeros_like(v); yaw = torch.zeros_like(v)
        rows = []
        for step in range(a.shape[-2]):
            ax, ay = a[..., step, 0], a[..., step, 1]
            kappa = ay / v.square().clamp_min(v_floor ** 2)
            yaw = yaw + v * kappa * dt
            v = (v + ax * dt).clamp(0, v_max)
            x = x + v * yaw.cos() * dt
            y = y + v * yaw.sin() * dt
            rows.append(torch.stack((x, y, yaw, v, ax, ay, kappa), -1))
        return torch.stack(rows, -2)


def euler_flow(velocity_fn, noise, steps=10):
    """Solver arithmetic is FP32; velocity_fn owns its BF16 network context."""
    with torch.autocast(device_type=noise.device.type, enabled=False):
        u = noise.float()
        for i in range(steps):
            s = torch.full(u.shape[:-2], i / steps, device=u.device, dtype=torch.float32)
            u = u + velocity_fn(u, s).float() / steps
        return u
