"""Input transform functions used by model paths."""
from __future__ import annotations

import torch


def local_span_mask(x: torch.Tensor, p_max: float = 0.4, c_max: int = 6) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply."""
    B, T, _ = x.shape
    mask = torch.zeros(B, T, dtype=torch.bool, device=x.device)
    for b in range(B):
        p = torch.rand(1).item() * p_max
        span = torch.randint(1, c_max + 1, ()).item()
        step = int(1 / p) if p > 0 else T + 1
        idxs = torch.arange(0, T - span, step)
        for t0 in idxs:
            mask[b, t0 : t0 + span] = True
    mask = mask.unsqueeze(-1).expand_as(x)
    xm = x.clone()
    xm[mask] = 0.0
    return xm, mask


def suffix_mask(x: torch.Tensor, past_len: int = 50) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask timesteps from onward."""
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[:, past_len:] = True
    xm = x.clone()
    xm[mask] = 0.0
    return xm, mask
