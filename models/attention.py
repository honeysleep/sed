import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SymmetricConfidenceBiasedMHA(nn.Module):
    def __init__(self, d_model, nhead=4, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.scale = math.sqrt(self.d_k)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # confidence → per-head bias (shared projection for Q/K side)
        self.conf_proj = nn.Linear(1, nhead)
        self.alpha = nn.Parameter(torch.tensor(0.0))  
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, confidence=None, return_weights=False):
        B, T, D = x.shape
        H, dk = self.nhead, self.d_k

        Q = self.q_proj(x).view(B, T, H, dk).transpose(1, 2)  # (B,H,T,dk)
        K = self.k_proj(x).view(B, T, H, dk).transpose(1, 2)
        V = self.v_proj(x).view(B, T, H, dk).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B,H,T,T)

        alpha = torch.sigmoid(self.alpha)
        if confidence is not None:
            cb = self.conf_proj(confidence.unsqueeze(-1))  # (B,T,H)
            cb = cb.permute(0, 2, 1)                       # (B,H,T)
            # Symmetric bias: (B,H,T,1) + (B,H,1,T) → broadcasts to (B,H,T,T)
            scores = scores + alpha * (cb.unsqueeze(-1) + cb.unsqueeze(-2))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(out)

        if return_weights:
            return out, attn, alpha.item()
        return out


class LearnableConfidencePooler(nn.Module):
    def __init__(self, n_mels: int = 40, n_resolutions: int = 3):
        super().__init__()
        self.freq_weights = nn.Parameter(torch.zeros(n_mels))
        self.res_weights = nn.Parameter(torch.zeros(n_resolutions))

    def forward(self, confidence_maps):
        """
        Args:
            confidence_maps: (B, n_res, n_mels, T)
        Returns:
            conf: (B, T)
        """
        freq_w = F.softmax(self.freq_weights, dim=0)  # (n_mels,)
        res_w = F.softmax(self.res_weights, dim=0)    # (n_res,)

        # Weighted sum over frequency: (B, n_res, T)
        conf = (confidence_maps * freq_w.view(1, 1, -1, 1)).sum(dim=2)
        # Weighted sum over resolution: (B, T)
        conf = (conf * res_w.view(1, -1, 1)).sum(dim=1)
        return conf
