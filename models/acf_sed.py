import torch
import torch.nn as nn
import torch.nn.functional as F

from .ale_frontend import MultiResolutionALEBank
from .attention import LearnableConfidencePooler
from .cnn_encoder import EventCNN, NoiseCNN
from .confidence_encoder_blocks import DualPathConfidenceEncoderBlock


class ACF_SED(nn.Module):
    def __init__(
        self,
        num_classes=3,
        n_mels=40,
        sample_rate=4000,
        d_model=128,
        nhead=4,
        num_layers=4,
        kernel_size=31,
        expansion_factor=4,
        dropout=0.3,
        n_fft=512,
        hop_length=40,
        win_length=100,
        drop_path_rate=0.1,
        ale_loss_weight=0.01,  
        frame_level=True,
        mu_mode='scalar',      # 'fixed' | 'scalar' |
        mu_init=0.05,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.d_model = d_model
        self.use_ale = True
        self.frame_level = True

        # 1. MRAB
        self.mrab = MultiResolutionALEBank(
            n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            sample_rate=sample_rate, n_mels=n_mels,
            mu_mode=mu_mode, mu_init=mu_init,
        )

        # 2. Dual-Stream CNN
        self.cnn = EventCNN(in_channels=3, d_model=d_model, dropout=dropout)
        self.noise_cnn = NoiseCNN(in_channels=3, d_model=d_model,
                                  dropout=dropout * 0.7)

        # 3. Learnable Confidence Pooler
        self.conf_pooler = LearnableConfidencePooler(
            n_mels=n_mels, n_resolutions=3
        )

        # 4. Dual-Path Confidence Encoder Blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        self.confidence_encoder_blocks = nn.ModuleList([
            DualPathConfidenceEncoderBlock(
                d_model=d_model,
                nhead=nhead,
                kernel_size=kernel_size,
                expansion_factor=expansion_factor,
                dropout=dropout,
                drop_path_rate=dpr[i],
            )
            for i in range(num_layers)
        ])

        # 5. Frame Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, labels=None, return_intermediates=False, **kwargs):
        # 1. MRAB
        enhanced_mels, noise_mels, confidence_maps = self.mrab(x)

        # 2. Dual-Stream CNN
        event_seq = self.cnn(enhanced_mels)     # (B, T', d_model)
        noise_seq = self.noise_cnn(noise_mels)  # (B, T', d_model)

        # 3. Confidence pooling & time-axis 
        conf = self.conf_pooler(confidence_maps)  # (B, T_mrab)
        T_seq = event_seq.size(1)
        if conf.size(1) != T_seq:
            conf = F.interpolate(
                conf.unsqueeze(1), size=T_seq, mode='linear', align_corners=False
            ).squeeze(1)

        if noise_seq.size(1) != T_seq:
            noise_seq = F.interpolate(
                noise_seq.transpose(1, 2), size=T_seq,
                mode='linear', align_corners=False
            ).transpose(1, 2)

        # 4. Dual-Path Confidence Encoder Blocks
        event, noise = event_seq, noise_seq
        layer_intermediates = []
        for block in self.confidence_encoder_blocks:
            if return_intermediates:
                event, noise, layer_info = block(event, noise, conf, return_intermediates=True)
                layer_intermediates.append(layer_info)
            else:
                event, noise = block(event, noise, conf)

        # 5. Frame Classifier
        output = self.classifier(event)  # (B, T', num_classes)

        if return_intermediates:
            intermediates = {
                'enhanced_mels':       enhanced_mels,
                'noise_mels':          noise_mels,
                'confidence_maps':     confidence_maps,
                'confidence':          conf,
                'event_seq':           event_seq,
                'event_features':      event,
                'layer_intermediates': layer_intermediates,
            }
            return output, intermediates

        return output

    def get_ale_loss(self):
        return torch.tensor(0.0)

    def get_ale_status(self):
        status = {}
        for i, f in enumerate(self.mrab.filters):
            mu = f.get_mu()
            if f.mu_mode == 'per_freq':
                status[f'filter_{i}_mu_mean'] = mu.mean().item()
                status[f'filter_{i}_mu_min']  = mu.min().item()
                status[f'filter_{i}_mu_max']  = mu.max().item()
            else:
                status[f'filter_{i}_mu'] = mu.item()
        return status

    def get_trainable_ale_params(self):
        return list(self.mrab.parameters())

    def get_non_ale_params(self):
        ale_ids = set(id(p) for p in self.mrab.parameters())
        return [p for p in self.parameters() if id(p) not in ale_ids]


def create_acf_sed(num_classes=3, n_mels=40, sample_rate=4000,
                   d_model=128, nhead=4, num_layers=4,
                   dropout=0.3, drop_path_rate=0.1,
                   frame_level=True, mu_mode='scalar', mu_init=0.05, **kwargs):

    model = ACF_SED(
        num_classes=num_classes,
        n_mels=n_mels,
        sample_rate=sample_rate,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout,
        drop_path_rate=drop_path_rate,
        frame_level=frame_level,
        mu_mode=mu_mode,
        mu_init=mu_init,
        **kwargs,
    )

    return model
