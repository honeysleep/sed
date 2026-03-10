import torch
import torch.nn as nn


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep_prob)
        return x * mask / keep_prob


class ALEAwareModulation(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Linear(1, d_model)
        self.temperature = nn.Parameter(torch.tensor(0.0))

    def forward(self, features, confidence):
        mod = torch.sigmoid(self.proj(confidence.unsqueeze(-1)))  # (B,T,d_model)
        tau = torch.sigmoid(self.temperature)
        return features * (1.0 + tau * mod)
