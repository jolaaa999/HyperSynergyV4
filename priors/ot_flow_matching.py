"""Optimal transport flow matching on support-adapted task latents."""
import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment


def squared_cost(x0, x1):
    if x0.dim() != 2 or x0.shape != x1.shape or x0.size(0) == 0:
        raise ValueError("OT inputs must have the same nonempty [B,d] shape")
    if x0.device != x1.device or x0.dtype != x1.dtype:
        raise ValueError("OT inputs must share device and dtype")
    return (x0[:, None, :] - x1[None, :, :]).square().sum(dim=-1)


def optimal_transport_pair(x0, x1):
    # Only assignment is nondifferentiable. Index the ORIGINAL target tensor.
    with torch.no_grad():
        cost = squared_cost(x0, x1)
        if not torch.isfinite(cost).all():
            raise ValueError("Non-finite OT cost")
        rows, cols = linear_sum_assignment(cost.cpu().double().numpy())
        rows = torch.as_tensor(rows, device=x0.device, dtype=torch.long)
        cols = torch.as_tensor(cols, device=x1.device, dtype=torch.long)
    return x0.index_select(0, rows), x1.index_select(0, cols), rows, cols


def flow_path(x0, x1, t):
    if t.shape != (x0.size(0), 1) or x0.shape != x1.shape:
        raise ValueError("Expected x0/x1 [B,d] and t [B,1]")
    if t.device != x0.device or t.dtype != x0.dtype:
        raise ValueError("Time must share source device and dtype")
    if not torch.isfinite(t).all() or ((t < 0) | (t > 1)).any():
        raise ValueError("Time must lie in [0,1]")
    return (1 - t) * x0 + t * x1, x1 - x0


class OTFlowMatchingPrior(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        if latent_dim != 128:
            raise ValueError("Approved velocity architecture requires latent_dim=128")
        self.latent_dim = latent_dim
        self.velocity = nn.Sequential(
            nn.Linear(129, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, 128),
        )

    def forward(self, h):
        if h.dim() != 2 or h.size(1) != self.latent_dim or h.size(0) == 0:
            raise ValueError("Expected nonempty adapted task latents [B,128]")
        if not torch.isfinite(h).all():
            raise ValueError("Non-finite adapted latents")
        x0 = torch.randn_like(h)
        x0, x1, _, _ = optimal_transport_pair(x0, h)
        t = torch.rand((h.size(0), 1), device=h.device, dtype=h.dtype)
        xt, target_velocity = flow_path(x0, x1, t)
        velocity = self.velocity(torch.cat((xt, t), dim=-1))
        return F.mse_loss(velocity, target_velocity, reduction="mean")
