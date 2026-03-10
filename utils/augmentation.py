import numpy as np
import torch


def apply_filter_aug(waveforms: torch.Tensor,
                     prob: float = 0.5,
                     max_bandwidth_ratio: float = 0.4,
                     num_bands: int = 2) -> torch.Tensor:
    if prob <= 0.0:
        return waveforms
    if torch.rand(1).item() > prob:
        return waveforms
    if waveforms.dim() != 2:
        return waveforms

    B, N = waveforms.shape
    spec = torch.fft.rfft(waveforms, dim=-1)
    F_bins = spec.shape[-1]
    mask = torch.ones_like(spec)

    for b in range(B):
        for _ in range(max(1, num_bands)):
            bandwidth = int(F_bins * (torch.rand(1).item() * max_bandwidth_ratio))
            if bandwidth <= 0:
                continue
            start = int((F_bins - bandwidth) * torch.rand(1).item())
            end = min(F_bins, start + bandwidth)
            mask[b, start:end] = 0.0

    spec_filtered = spec * mask
    return torch.fft.irfft(spec_filtered, n=N, dim=-1)


def apply_balanced_mixup(x: torch.Tensor,
                         y: torch.Tensor,
                         alpha: float = 0.4):
    if alpha <= 0.0 or x.size(0) < 2:
        return x, y

    batch_size = x.size(0)
    lam = np.random.beta(alpha, alpha)

    has_event_indices = torch.where(y.sum(dim=1) > 0)[0]
    if len(has_event_indices) > 0:
        perm_indices = has_event_indices[
            torch.randint(len(has_event_indices), (batch_size,), device=x.device)
        ]
    else:
        perm_indices = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1.0 - lam) * x[perm_indices]
    mixed_y = lam * y + (1.0 - lam) * y[perm_indices]
    return mixed_x, mixed_y
