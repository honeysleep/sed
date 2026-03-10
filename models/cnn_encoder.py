import torch.nn as nn


class EventCNN(nn.Module):
    def __init__(self, in_channels=3, d_model=128, dropout=0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32), nn.GELU(),
            nn.MaxPool2d((2, 1)), nn.Dropout2d(dropout * 0.5),

            nn.Conv2d(32, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64), nn.GELU(),
            nn.MaxPool2d((2, 1)), nn.Dropout2d(dropout * 0.5),

            nn.Conv2d(64, d_model, 3, 1, 1, bias=False),
            nn.BatchNorm2d(d_model), nn.GELU(),
            nn.MaxPool2d((2, 1)), nn.Dropout2d(dropout),
        )
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))

    def forward(self, enhanced_mels):
        """
        Args:
            enhanced_mels: (B, 3, n_mels, T)
        Returns:
            seq: (B, T', d_model)
        """
        x = self.cnn(enhanced_mels)       # (B, d_model, F', T')
        x = self.freq_pool(x).squeeze(2)  # (B, d_model, T')
        return x.transpose(1, 2)          # (B, T', d_model)


class NoiseCNN(nn.Module):
    def __init__(self, in_channels=3, d_model=128, dropout=0.2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32), nn.GELU(),
            nn.MaxPool2d((2, 1)),

            nn.Conv2d(32, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64), nn.GELU(),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(dropout),
        )
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))
        self.proj = nn.Linear(64, d_model)

    def forward(self, noise_mels):
        x = self.cnn(noise_mels)          # (B, 64, F', T')
        x = self.freq_pool(x).squeeze(2)  # (B, 64, T')
        x = x.transpose(1, 2)            # (B, T', 64)
        return self.proj(x)               # (B, T', d_model)
