import math
import torch
import torch.nn as nn
import librosa


class NLMSFilter(nn.Module):
    MU_MODES = ('fixed', 'scalar', 'per_freq')

    def __init__(self, num_freq_bins, filter_order=15, delay=3,
                 mu_init=0.05, eps=1e-8, mu_mode='scalar'):
        super().__init__()
        assert mu_mode in self.MU_MODES, \
            f"mu_mode must be one of {self.MU_MODES}, got '{mu_mode}'"

        self.num_freq_bins = num_freq_bins
        self.filter_order = filter_order
        self.delay = delay
        self.eps = eps
        self.mu_mode = mu_mode

        init_logit = math.log(mu_init / (1.0 - mu_init + 1e-8))

        if mu_mode == 'fixed':
            self.register_buffer('mu_val', torch.tensor(float(mu_init)))
        elif mu_mode == 'scalar':
            self.mu_logit = nn.Parameter(torch.tensor(init_logit))
        else:  # per_freq
            self.mu_logit = nn.Parameter(
                torch.full((num_freq_bins,), init_logit)
            )

        # Initial filter coefficients
        self.register_buffer('h_init_real',
                             torch.zeros(num_freq_bins, filter_order))
        self.register_buffer('h_init_imag',
                             torch.zeros(num_freq_bins, filter_order))

    def get_mu(self):
        if self.mu_mode == 'fixed':
            return self.mu_val
        return torch.sigmoid(self.mu_logit)

    def forward(self, Y):
        """
        Args:
            Y: (B, K, M) complex STFT tensor
        Returns:
            E: (B, K, M) enhanced signal (error)
            Z: (B, K, M) noise estimate (predicted)
        """
        B, K, M = Y.shape
        L = self.filter_order
        tau = self.delay
        mu = self.get_mu()
        eps = self.eps
        device = Y.device

        h = torch.complex(
            self.h_init_real.unsqueeze(0).expand(B, -1, -1).clone(),
            self.h_init_imag.unsqueeze(0).expand(B, -1, -1).clone()
        ).to(device)

        Z = torch.zeros_like(Y)
        E = torch.zeros_like(Y)

        start_frame = tau + L - 1

        if self.mu_mode == 'per_freq':
            mu_eff = mu.view(1, K, 1)
        else:
            mu_eff = mu

        for m in range(start_frame, M):
            indices = [m - tau - l for l in range(L)]
            y_delayed = torch.stack([Y[:, :, idx] for idx in indices], dim=-1)

            z_m = torch.sum(torch.conj(h) * y_delayed, dim=-1)
            Z[:, :, m] = z_m

            e_m = Y[:, :, m] - z_m
            E[:, :, m] = e_m

            power = torch.sum(torch.abs(y_delayed) ** 2, dim=-1,
                              keepdim=True) + eps
            delta_h = mu_eff * torch.conj(e_m).unsqueeze(-1) * y_delayed / power
            h = h + delta_h

        for m in range(start_frame):
            E[:, :, m] = Y[:, :, m]

        return E, Z


class MultiResolutionALEBank(nn.Module):
    """
    Multi-Resolution ALE Bank (MRAB).

    Shared STFT -> 3 parallel NLMS filters with different delay/filter_order
    -> mel conversion -> enhanced_mels, noise_mels, confidence_maps.

    Output shapes: (B, 3, n_mels, T) for each of the three outputs.
    """

    def __init__(self, n_fft=512, hop_length=40, win_length=100,
                 sample_rate=4000, n_mels=40, eps=1e-8,
                 mu_mode='scalar', mu_init=0.05):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.eps = eps
        num_freq_bins = n_fft // 2 + 1

        self.register_buffer('window', torch.hamming_window(win_length))

        # 3 NLMS filters at different resolutions: (τ=1,L=8), (τ=3,L=15), (τ=6,L=24)
        self.filters = nn.ModuleList([
            NLMSFilter(num_freq_bins, filter_order=8,  delay=1, mu_init=mu_init, mu_mode=mu_mode),
            NLMSFilter(num_freq_bins, filter_order=15, delay=3, mu_init=mu_init, mu_mode=mu_mode),
            NLMSFilter(num_freq_bins, filter_order=24, delay=6, mu_init=mu_init, mu_mode=mu_mode),
        ])

        # Shared mel filterbank
        mel_basis = librosa.filters.mel(
            sr=sample_rate, n_fft=n_fft, n_mels=n_mels,
            fmin=0, fmax=sample_rate // 2
        )
        self.register_buffer('mel_basis', torch.FloatTensor(mel_basis))

    def _stft(self, x):
        return torch.stft(
            x, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.win_length, window=self.window.to(x.device),
            return_complex=True, center=True
        )

    def _to_mel(self, stft_complex):
        """Convert complex STFT to log-mel: (B, K, T) -> (B, n_mels, T)"""
        mag = torch.abs(stft_complex)
        mel = torch.matmul(self.mel_basis, mag)
        return torch.log(mel + 1e-9)

    def forward(self, x):
        Y = self._stft(x)  # (B, K, M) - shared STFT

        enhanced_list = []
        noise_list = []
        conf_list = []

        for nlms in self.filters:
            E, Z = nlms(Y)  # (B, K, M) each

            e_mel = self._to_mel(E)  # (B, n_mels, T)
            z_mel = self._to_mel(Z)

            # Confidence map: |E|^2 / (|E|^2 + |Z|^2 + eps)
            e_power = torch.abs(E) ** 2
            z_power = torch.abs(Z) ** 2
            conf_stft = e_power / (e_power + z_power + self.eps)
            conf_mel = torch.matmul(self.mel_basis, conf_stft)
            # Normalize to [0, 1]
            conf_mel = conf_mel / (conf_mel.max(dim=-1, keepdim=True)[0].max(
                dim=-2, keepdim=True)[0] + self.eps)

            enhanced_list.append(e_mel)
            noise_list.append(z_mel)
            conf_list.append(conf_mel)

        # Stack: (B, 3, n_mels, T)
        enhanced_mels = torch.stack(enhanced_list, dim=1)
        noise_mels = torch.stack(noise_list, dim=1)
        confidence_maps = torch.stack(conf_list, dim=1)

        return enhanced_mels, noise_mels, confidence_maps
