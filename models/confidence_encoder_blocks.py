import torch
import torch.nn as nn

from .attention import SymmetricConfidenceBiasedMHA
from .modulation import ALEAwareModulation, DropPath


class _FeedForward(nn.Module):
    def __init__(self, d_model, expansion_factor=4, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_model * expansion_factor)
        self.activation = nn.SiLU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * expansion_factor, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        return self.dropout2(x)


class _LightweightFFN(nn.Module):
    def __init__(self, d_model, expansion_factor=2, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_model * expansion_factor)
        self.activation = nn.SiLU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * expansion_factor, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        return self.dropout2(x)


class _ConvModule(nn.Module):
    def __init__(self, d_model, kernel_size=31, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Linear(d_model, d_model * 2)
        self.glu = nn.GLU(dim=-1)
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel_size,
            padding=(kernel_size - 1) // 2, groups=d_model
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = x.transpose(1, 2)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = x.transpose(1, 2)
        x = self.pointwise_conv2(x)
        return self.dropout(x)


class DualPathConfidenceEncoderBlock(nn.Module):
    def __init__(self, d_model, nhead=4, kernel_size=31,
                 expansion_factor=4, dropout=0.1, drop_path_rate=0.0):
        super().__init__()

        # ---- Event Path ----
        self.ffn1 = _FeedForward(d_model, expansion_factor, dropout)

        self.attn_norm = nn.LayerNorm(d_model)
        self.cb_mha = SymmetricConfidenceBiasedMHA(d_model, nhead, dropout)
        self.attn_drop = nn.Dropout(dropout)

        self.conv = _ConvModule(d_model, kernel_size, dropout)

        self.ffn2 = _FeedForward(d_model, expansion_factor, dropout)
        self.norm = nn.LayerNorm(d_model)

        # ---- Noise Lightweight Path ----
        self.ffn_noise = _LightweightFFN(d_model, expansion_factor=2, dropout=dropout)
        self.norm_noise = nn.LayerNorm(d_model)

        # ---- Cross-Attention: event(Q) ← noise(K,V) ----
        cross_nhead = max(1, nhead // 2)
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads=cross_nhead,
            dropout=dropout, batch_first=True
        )
        self.cross_q_norm = nn.LayerNorm(d_model)
        self.cross_kv_norm = nn.LayerNorm(d_model)
        self.cross_drop = nn.Dropout(dropout)

        # ---- Gated Fusion ----
        self.gate_proj = nn.Linear(d_model * 2, d_model)

        # ---- AFM ----
        self.afm = ALEAwareModulation(d_model)

        # ---- DropPath ----
        self.drop_path = DropPath(drop_path_rate)

    def forward(self, event, noise, confidence, return_intermediates=False):
        # ---- Event Path ----
        event = event + 0.5 * self.drop_path(self.ffn1(event))

        if return_intermediates:
            attn_out, attn_weights, cb_alpha = self.cb_mha(
                self.attn_norm(event), confidence, return_weights=True
            )
            event = event + self.drop_path(self.attn_drop(attn_out))
        else:
            event = event + self.drop_path(
                self.attn_drop(self.cb_mha(self.attn_norm(event), confidence))
            )

        event = event + self.drop_path(self.conv(event))
        event = event + 0.5 * self.drop_path(self.ffn2(event))
        event = self.norm(event)

        # ---- Noise Lightweight Path ----
        noise = noise + self.ffn_noise(noise)
        noise = self.norm_noise(noise)

        # ---- Cross-Attention Fusion ----
        noise_normed = self.cross_kv_norm(noise)
        cross_out, _ = self.cross_attn(
            query=self.cross_q_norm(event),
            key=noise_normed,
            value=noise_normed,
        )
        cross_out = self.cross_drop(cross_out)

        # ---- Gated Fusion ----
        gate = torch.sigmoid(
            self.gate_proj(torch.cat([event, cross_out], dim=-1))
        )  # (B, T, d_model)
        event = event * gate + cross_out * (1.0 - gate)

        # ---- AFM ----
        event = self.afm(event, confidence)

        if return_intermediates:
            layer_info = {
                'attn_weights': attn_weights,  # (B, H, T, T)
                'gate_values':  gate,          # (B, T, d_model)
                'cb_alpha':     cb_alpha,      # float
            }
            return event, noise, layer_info

        return event, noise
